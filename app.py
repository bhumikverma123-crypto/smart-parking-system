"""
Smart Parking System — Flask Backend
=====================================
Mirrors every feature from the frontend:
  • 20×20 grid, 32 parking spots in four 2×4 clusters
  • A* pathfinding through corridor cells only (spots are obstacles)
  • Car queue  → process-next / process-all
  • Manual spot selection with path preview
  • Reservations (Premium) with 10-minute auto-expiry
  • Undo / Redo stack (up to 30 snapshots)
  • Full activity log
  • Reset

Run:
    python app.py
    # API listens on http://localhost:5000
"""

from __future__ import annotations

import copy
import heapq
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional
from flask import Flask, jsonify, request, Response

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

GRID         = 20
ENTRANCE     = (10, 0)        # (col, row)
MAX_UNDO     = 30
RESERVE_SECS = 10 * 60        # 10 minutes

SPOT_CONFIGS: list[tuple[int, int]] = [
    # Left-top cluster
    (2, 4), (3, 4), (4, 4), (5, 4),
    (2, 5), (3, 5), (4, 5), (5, 5),
    # Left-bottom cluster
    (2, 10), (3, 10), (4, 10), (5, 10),
    (2, 11), (3, 11), (4, 11), (5, 11),
    # Right-top cluster
    (15, 4), (16, 4), (17, 4), (18, 4),
    (15, 5), (16, 5), (17, 5), (18, 5),
    # Right-bottom cluster
    (15, 10), (16, 10), (17, 10), (18, 10),
    (15, 11), (16, 11), (17, 11), (18, 11),
]

EMOJIS = ["🚗", "🚙", "🚕", "🚐", "🏎️"]

# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class Spot:
    id:       int
    x:        int
    y:        int
    occupied: bool = False
    car_id:   Optional[int] = None


@dataclass
class Car:
    id:    int
    emoji: str


@dataclass
class ParkedCar:
    id:      int
    emoji:   str
    spot_id: int


@dataclass
class Reservation:
    spot_id: int
    expires: float          # Unix timestamp


@dataclass
class LogEntry:
    timestamp: str
    message:   str
    level:     str          # ok | warn | err | info


@dataclass
class Snapshot:
    """Full serialisable state for undo / redo."""
    label:        str
    spots:        list[dict]
    car_counter:  int
    car_queue:    list[dict]
    parked_cars:  list[dict]
    last_path:    list[list[int]]
    reservations: dict[str, dict]


# ─────────────────────────────────────────────
#  A* Pathfinder
# ─────────────────────────────────────────────

def astar(
    spots: list[Spot],
    sx: int, sy: int,
    ex: int, ey: int,
) -> Optional[list[tuple[int, int]]]:
    """
    Find the shortest corridor path from (sx,sy) to (ex,ey).

    Rules:
      • Every parking-spot cell is an obstacle EXCEPT the destination.
      • Movement is 4-directional (no diagonals).
      • Returns a list of (col, row) tuples including start and end,
        or None if no path exists.
    """
    # Build obstacle set: all spot cells except the destination
    obstacles: set[tuple[int, int]] = set()
    for s in spots:
        if not (s.x == ex and s.y == ey):
            obstacles.add((s.x, s.y))

    def h(x: int, y: int) -> int:
        return abs(x - ex) + abs(y - ey)

    # (f, g, x, y)
    open_heap: list[tuple[int, int, int, int]] = []
    heapq.heappush(open_heap, (h(sx, sy), 0, sx, sy))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score:   dict[tuple[int, int], int]             = {(sx, sy): 0}

    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while open_heap:
        _, g_cur, cx, cy = heapq.heappop(open_heap)

        if (cx, cy) == (ex, ey):
            # Reconstruct path
            path: list[tuple[int, int]] = []
            node = (cx, cy)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path

        # Skip stale heap entries
        if g_cur > g_score.get((cx, cy), float("inf")):
            continue

        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                continue
            if (nx, ny) in obstacles:
                continue

            new_g = g_cur + 1
            if new_g < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = new_g
                came_from[(nx, ny)] = (cx, cy)
                f = new_g + h(nx, ny)
                heapq.heappush(open_heap, (f, new_g, nx, ny))

    return None  # unreachable


# ─────────────────────────────────────────────
#  Parking Lot State Machine
# ─────────────────────────────────────────────

class ParkingLot:
    """
    Thread-safe in-memory state for the entire parking system.
    All public methods acquire self._lock before mutating state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_state()
        self._undo_stack: list[Snapshot] = []
        self._redo_stack: list[Snapshot] = []
        self._log:        list[LogEntry] = []
        self._reserve_timers: dict[int, threading.Timer] = {}
        self._log_entry("System online — 32 spots ready", "ok")

    # ── internal helpers ──────────────────────

    def _reset_state(self) -> None:
        self.spots:       list[Spot]       = [
            Spot(id=i + 1, x=x, y=y)
            for i, (x, y) in enumerate(SPOT_CONFIGS)
        ]
        self.car_counter: int              = 0
        self.car_queue:   list[Car]        = []
        self.parked_cars: list[ParkedCar]  = []
        self.last_path:   list[list[int]]  = []
        self.reservations: dict[int, Reservation] = {}

    def _log_entry(self, message: str, level: str = "info") -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._log.insert(0, LogEntry(timestamp=ts, message=message, level=level))
        if len(self._log) > 200:
            self._log = self._log[:200]

    def _spot_by_id(self, spot_id: int) -> Optional[Spot]:
        return next((s for s in self.spots if s.id == spot_id), None)

    def _find_nearest(self) -> Optional[Spot]:
        """Return the free spot with the shortest A* corridor path."""
        best: Optional[Spot] = None
        best_len = float("inf")
        for s in self.spots:
            if s.occupied:
                continue
            if s.id in self.reservations:
                continue
            path = astar(self.spots, ENTRANCE[0], ENTRANCE[1], s.x, s.y)
            if path and len(path) < best_len:
                best_len = len(path)
                best = s
        return best

    # ── snapshot helpers ──────────────────────

    def _capture(self, label: str) -> Snapshot:
        return Snapshot(
            label        = label,
            spots        = [asdict(s) for s in self.spots],
            car_counter  = self.car_counter,
            car_queue    = [asdict(c) for c in self.car_queue],
            parked_cars  = [asdict(p) for p in self.parked_cars],
            last_path    = copy.deepcopy(self.last_path),
            reservations = {
                str(k): {"spot_id": v.spot_id, "expires": v.expires}
                for k, v in self.reservations.items()
            },
        )

    def _push_undo(self, label: str) -> None:
        self._undo_stack.append(self._capture(label))
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _apply_snapshot(self, snap: Snapshot) -> None:
        # Cancel all live reservation timers
        for t in self._reserve_timers.values():
            t.cancel()
        self._reserve_timers.clear()

        self.spots = [
            Spot(**s) for s in snap.spots
        ]
        self.car_counter = snap.car_counter
        self.car_queue   = [Car(**c) for c in snap.car_queue]
        self.parked_cars = [ParkedCar(**p) for p in snap.parked_cars]
        self.last_path   = snap.last_path
        self.reservations = {}

        now = time.time()
        for sid_str, rv in snap.reservations.items():
            sid = int(sid_str)
            remaining = rv["expires"] - now
            if remaining > 0:
                self.reservations[sid] = Reservation(
                    spot_id=rv["spot_id"],
                    expires=rv["expires"],
                )
                self._arm_reservation_timer(sid, remaining)
            # else: already expired — skip

    def _arm_reservation_timer(self, spot_id: int, delay: float) -> None:
        """Fire an auto-expiry timer for a reservation."""
        def expire():
            with self._lock:
                if spot_id in self.reservations:
                    del self.reservations[spot_id]
                    self._reserve_timers.pop(spot_id, None)
                    self._log_entry(
                        f"Reservation for Spot #{spot_id} expired", "warn"
                    )
        timer = threading.Timer(delay, expire)
        timer.daemon = True
        timer.start()
        self._reserve_timers[spot_id] = timer

    # ── serialisation ─────────────────────────

    def _to_dict(self) -> dict:
        now = time.time()
        return {
            "grid":        GRID,
            "entrance":    list(ENTRANCE),
            "spots":       [asdict(s) for s in self.spots],
            "car_counter": self.car_counter,
            "car_queue":   [asdict(c) for c in self.car_queue],
            "parked_cars": [asdict(p) for p in self.parked_cars],
            "last_path":   self.last_path,
            "reservations": {
                str(k): {
                    "spot_id":    v.spot_id,
                    "expires":    v.expires,
                    "remaining_s": max(0, round(v.expires - now)),
                }
                for k, v in self.reservations.items()
            },
            "stats": {
                "total":    len(self.spots),
                "available": sum(1 for s in self.spots if not s.occupied),
                "occupied":  sum(1 for s in self.spots if s.occupied),
                "queued":    len(self.car_queue),
                "reserved":  len(self.reservations),
            },
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "undo_label": self._undo_stack[-1].label if self._undo_stack else None,
            "log": [asdict(e) for e in self._log[:50]],
        }

    # ── public API methods ────────────────────

    def get_state(self) -> dict:
        with self._lock:
            return self._to_dict()

    def add_car(self) -> dict:
        with self._lock:
            self._push_undo(f"Add Car #{self.car_counter + 1} to queue")
            self.car_counter += 1
            emoji = EMOJIS[(self.car_counter - 1) % len(EMOJIS)]
            car = Car(id=self.car_counter, emoji=emoji)
            self.car_queue.append(car)
            self._log_entry(
                f"Car #{car.id} {car.emoji} joined the queue", "info"
            )
            return self._to_dict()

    def process_next(self, spot_id: Optional[int] = None) -> dict:
        with self._lock:
            if not self.car_queue:
                return {"error": "Queue is empty — add a car first", **self._to_dict()}

            car = self.car_queue[0]   # peek first

            # Resolve target
            if spot_id is not None:
                target = self._spot_by_id(spot_id)
                if target is None:
                    return {"error": f"Spot #{spot_id} does not exist", **self._to_dict()}
            else:
                target = self._find_nearest()

            if target is None:
                return {"error": "No reachable spots available", **self._to_dict()}

            if target.occupied:
                return {"error": f"Spot #{target.id} is already occupied", **self._to_dict()}

            if target.id in self.reservations:
                rv = self.reservations[target.id]
                remaining = max(0, round(rv.expires - time.time()))
                return {
                    "error": f"Spot #{target.id} is reserved — locked for {remaining}s",
                    **self._to_dict(),
                }

            # Compute A* path
            path = astar(self.spots, ENTRANCE[0], ENTRANCE[1], target.x, target.y)
            if path is None:
                return {
                    "error": f"No corridor path to Spot #{target.id}",
                    **self._to_dict(),
                }

            # Commit
            self._push_undo(f"Park Car #{car.id} at Spot #{target.id}")
            self.car_queue.pop(0)

            target.occupied = True
            target.car_id   = car.id

            # Clear any reservation on this spot
            if target.id in self.reservations:
                t = self._reserve_timers.pop(target.id, None)
                if t:
                    t.cancel()
                del self.reservations[target.id]

            self.last_path = [list(p) for p in path]
            self.parked_cars.append(
                ParkedCar(id=car.id, emoji=car.emoji, spot_id=target.id)
            )

            steps = len(path) - 1
            self._log_entry(
                f"Car #{car.id} {car.emoji} → Spot #{target.id}  [{steps} steps]", "ok"
            )
            result = self._to_dict()
            result["path_info"] = {
                "spot_id": target.id,
                "spot_x":  target.x,
                "spot_y":  target.y,
                "steps":   steps,
                "cells":   len(path),
                "path":    self.last_path,
            }
            return result

    def process_all(self) -> dict:
        """Park every car currently in the queue (auto-assign nearest)."""
        with self._lock:
            parked = 0
            errors = []
            while self.car_queue:
                car  = self.car_queue[0]
                tgt  = self._find_nearest()
                if tgt is None:
                    errors.append(f"No spots for Car #{car.id}")
                    break
                path = astar(self.spots, ENTRANCE[0], ENTRANCE[1], tgt.x, tgt.y)
                if path is None:
                    errors.append(f"No path to Spot #{tgt.id} for Car #{car.id}")
                    break
                self._push_undo(f"Park Car #{car.id} at Spot #{tgt.id}")
                self.car_queue.pop(0)
                tgt.occupied = True
                tgt.car_id   = car.id
                if tgt.id in self.reservations:
                    t = self._reserve_timers.pop(tgt.id, None)
                    if t:
                        t.cancel()
                    del self.reservations[tgt.id]
                self.last_path = [list(p) for p in path]
                self.parked_cars.append(
                    ParkedCar(id=car.id, emoji=car.emoji, spot_id=tgt.id)
                )
                steps = len(path) - 1
                self._log_entry(
                    f"Car #{car.id} {car.emoji} → Spot #{tgt.id}  [{steps} steps]", "ok"
                )
                parked += 1
            result = self._to_dict()
            result["parked_count"] = parked
            if errors:
                result["errors"] = errors
            return result

    def preview_path(self, spot_id: int) -> dict:
        """
        Return the A* path to a spot WITHOUT parking a car.
        Used by the frontend's manual-select preview.
        """
        with self._lock:
            spot = self._spot_by_id(spot_id)
            if spot is None:
                return {"error": f"Spot #{spot_id} does not exist"}
            if spot.occupied:
                return {"error": f"Spot #{spot_id} is occupied"}
            if spot_id in self.reservations:
                rv = self.reservations[spot_id]
                remaining = max(0, round(rv.expires - time.time()))
                return {
                    "error": f"Spot #{spot_id} is reserved — locked for {remaining}s"
                }
            path = astar(self.spots, ENTRANCE[0], ENTRANCE[1], spot.x, spot.y)
            if path is None:
                return {"error": f"No corridor path to Spot #{spot_id}"}
            steps = len(path) - 1
            return {
                "spot_id": spot_id,
                "spot_x":  spot.x,
                "spot_y":  spot.y,
                "steps":   steps,
                "cells":   len(path),
                "path":    [list(p) for p in path],
            }

    def reserve_spot(self, spot_id: int) -> dict:
        with self._lock:
            spot = self._spot_by_id(spot_id)
            if spot is None:
                return {"error": f"Spot #{spot_id} does not exist", **self._to_dict()}
            if spot.occupied:
                return {"error": f"Spot #{spot_id} is already occupied", **self._to_dict()}
            if spot_id in self.reservations:
                return {"error": f"Spot #{spot_id} is already reserved", **self._to_dict()}

            self._push_undo(f"Reserve Spot #{spot_id}")
            expires = time.time() + RESERVE_SECS
            self.reservations[spot_id] = Reservation(spot_id=spot_id, expires=expires)
            self._arm_reservation_timer(spot_id, RESERVE_SECS)
            self._log_entry(f"Spot #{spot_id} reserved for 10 min", "ok")
            return self._to_dict()

    def cancel_reservation(self, spot_id: int) -> dict:
        with self._lock:
            if spot_id not in self.reservations:
                return {"error": f"No active reservation for Spot #{spot_id}", **self._to_dict()}
            self._push_undo(f"Cancel reservation Spot #{spot_id}")
            t = self._reserve_timers.pop(spot_id, None)
            if t:
                t.cancel()
            del self.reservations[spot_id]
            self._log_entry(f"Reservation for Spot #{spot_id} cancelled", "warn")
            return self._to_dict()

    def undo(self) -> dict:
        with self._lock:
            if not self._undo_stack:
                return {"error": "Nothing to undo", **self._to_dict()}
            self._redo_stack.append(self._capture("redo"))
            snap = self._undo_stack.pop()
            self._apply_snapshot(snap)
            self._log_entry(f"↩ Undone: {snap.label}", "warn")
            result = self._to_dict()
            result["undone_label"] = snap.label
            return result

    def redo(self) -> dict:
        with self._lock:
            if not self._redo_stack:
                return {"error": "Nothing to redo", **self._to_dict()}
            self._undo_stack.append(self._capture("before redo"))
            snap = self._redo_stack.pop()
            self._apply_snapshot(snap)
            self._log_entry("↪ Redone", "info")
            return self._to_dict()

    def reset(self) -> dict:
        with self._lock:
            for t in self._reserve_timers.values():
                t.cancel()
            self._reserve_timers.clear()
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._reset_state()
            self._log_entry("Parking lot reset", "info")
            return self._to_dict()

    def get_log(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._log[:limit]]


# ─────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────

app = Flask(__name__)
lot = ParkingLot()


def cors(response: Response) -> Response:
    """Add CORS headers to every response (no flask-cors dependency)."""
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.after_request
def after_request(response: Response) -> Response:
    return cors(response)


@app.route("/", methods=["OPTIONS", "GET", "POST", "DELETE"])
def handle_options():
    return jsonify({}), 200


# ── Utility ───────────────────────────────────

def ok(data: dict, status: int = 200):
    data["success"] = "error" not in data
    return jsonify(data), status


def err(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


# ── Routes ────────────────────────────────────

@app.route("/api/state", methods=["GET"])
def get_state():
    """
    GET /api/state
    Returns the complete current state of the parking lot.
    """
    return ok(lot.get_state())


@app.route("/api/cars", methods=["POST"])
def add_car():
    """
    POST /api/cars
    Add a new car to the queue.

    Response: full state
    """
    return ok(lot.add_car())


@app.route("/api/cars/process", methods=["POST"])
def process_next():
    """
    POST /api/cars/process
    Park the next car in the queue.

    Body (optional JSON):
        { "spot_id": 5 }   — manually assign to Spot #5
        {}                 — auto-assign nearest spot

    Response: full state + path_info
    """
    body     = request.get_json(silent=True) or {}
    spot_id  = body.get("spot_id")
    if spot_id is not None:
        try:
            spot_id = int(spot_id)
        except (TypeError, ValueError):
            return err("spot_id must be an integer")
    return ok(lot.process_next(spot_id=spot_id))


@app.route("/api/cars/process-all", methods=["POST"])
def process_all():
    """
    POST /api/cars/process-all
    Park every car in the queue, auto-assigning each to the nearest free spot.

    Response: full state + parked_count
    """
    return ok(lot.process_all())


@app.route("/api/spots/<int:spot_id>/path", methods=["GET"])
def preview_path(spot_id: int):
    """
    GET /api/spots/<spot_id>/path
    Preview the shortest A* corridor path to a spot without parking a car.
    Useful for the frontend's manual-select preview.

    Response:
        {
          "spot_id": 5,
          "spot_x": 3, "spot_y": 4,
          "steps": 12,
          "cells": 13,
          "path": [[10,0],[10,1],...]
        }
    """
    result = lot.preview_path(spot_id)
    if "error" in result:
        return ok(result, 400)
    return ok(result)


@app.route("/api/spots/<int:spot_id>/reserve", methods=["POST"])
def reserve_spot(spot_id: int):
    """
    POST /api/spots/<spot_id>/reserve
    Reserve a spot for 10 minutes (Premium feature).

    Response: full state
    """
    return ok(lot.reserve_spot(spot_id))


@app.route("/api/spots/<int:spot_id>/reserve", methods=["DELETE"])
def cancel_reservation(spot_id: int):
    """
    DELETE /api/spots/<spot_id>/reserve
    Cancel an active reservation.

    Response: full state
    """
    return ok(lot.cancel_reservation(spot_id))


@app.route("/api/undo", methods=["POST"])
def undo():
    """
    POST /api/undo
    Revert the last undoable action.

    Response: full state + undone_label
    """
    return ok(lot.undo())


@app.route("/api/redo", methods=["POST"])
def redo():
    """
    POST /api/redo
    Re-apply the last undone action.

    Response: full state
    """
    return ok(lot.redo())


@app.route("/api/reset", methods=["POST"])
def reset():
    """
    POST /api/reset
    Clear all state and return to initial conditions.
    Undo/redo history is also cleared.

    Response: full state
    """
    return ok(lot.reset())


@app.route("/api/log", methods=["GET"])
def get_log():
    """
    GET /api/log?limit=50
    Return the activity log (most recent first).
    """
    limit = request.args.get("limit", 100, type=int)
    return jsonify({"success": True, "log": lot.get_log(limit)})


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║     Smart Parking System  — Backend      ║")
    print("╠══════════════════════════════════════════╣")
    print("║  http://localhost:5000                   ║")
    print("║                                          ║")
    print("║  Endpoints:                              ║")
    print("║   GET  /api/state                        ║")
    print("║   POST /api/cars                         ║")
    print("║   POST /api/cars/process                 ║")
    print("║   POST /api/cars/process-all             ║")
    print("║   GET  /api/spots/<id>/path              ║")
    print("║   POST /api/spots/<id>/reserve           ║")
    print("║   DEL  /api/spots/<id>/reserve           ║")
    print("║   POST /api/undo                         ║")
    print("║   POST /api/redo                         ║")
    print("║   POST /api/reset                        ║")
    print("║   GET  /api/log                          ║")
    print("╚══════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
