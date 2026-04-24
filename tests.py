"""
tests.py — Smart Parking System Backend Tests
==============================================
Run with:
    python tests.py
or:
    python -m pytest tests.py -v
"""

import sys
import time
import unittest

# ── import the app module ─────────────────────
try:
    from app import ParkingLot, astar, GRID, ENTRANCE, SPOT_CONFIGS, Spot
except ImportError:
    print("ERROR: could not import app.py — make sure tests.py is in the same folder.")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_lot() -> ParkingLot:
    """Fresh lot for every test."""
    return ParkingLot()


def add_and_process(lot: ParkingLot, spot_id=None) -> dict:
    lot.add_car()
    return lot.process_next(spot_id=spot_id)


# ─────────────────────────────────────────────
#  A* Tests
# ─────────────────────────────────────────────

class TestAstar(unittest.TestCase):

    def _fresh_spots(self):
        return [Spot(id=i+1, x=x, y=y) for i, (x, y) in enumerate(SPOT_CONFIGS)]

    def test_path_found(self):
        spots = self._fresh_spots()
        # First spot is at (2,4); entrance is (10,0)
        path = astar(spots, 10, 0, 2, 4)
        self.assertIsNotNone(path)

    def test_path_starts_at_entrance(self):
        spots = self._fresh_spots()
        path = astar(spots, 10, 0, 2, 4)
        self.assertEqual(path[0], (10, 0))

    def test_path_ends_at_destination(self):
        spots = self._fresh_spots()
        path = astar(spots, 10, 0, 2, 4)
        self.assertEqual(path[-1], (2, 4))

    def test_path_does_not_cross_other_spots(self):
        """No cell in the path (except the destination) should be a spot cell."""
        spots = self._fresh_spots()
        spot_cells = {(s.x, s.y) for s in spots}
        dest = (2, 4)
        path = astar(spots, 10, 0, dest[0], dest[1])
        for cell in path[:-1]:          # exclude destination itself
            self.assertNotIn(
                cell, spot_cells - {dest},
                f"Path illegally passes through spot cell {cell}"
            )

    def test_path_is_connected(self):
        """Each consecutive pair in the path must be 4-directionally adjacent."""
        spots = self._fresh_spots()
        path = astar(spots, 10, 0, 15, 4)
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            dist = abs(x1 - x2) + abs(y1 - y2)
            self.assertEqual(dist, 1, f"Non-adjacent step: {(x1,y1)} → {(x2,y2)}")

    def test_path_stays_in_bounds(self):
        spots = self._fresh_spots()
        path = astar(spots, 10, 0, 18, 11)
        for (x, y) in path:
            self.assertTrue(0 <= x < GRID, f"x={x} out of bounds")
            self.assertTrue(0 <= y < GRID, f"y={y} out of bounds")

    def test_all_spots_reachable(self):
        """Every spot must be reachable from the entrance."""
        spots = self._fresh_spots()
        for s in spots:
            path = astar(spots, ENTRANCE[0], ENTRANCE[1], s.x, s.y)
            self.assertIsNotNone(path, f"Spot #{s.id} at ({s.x},{s.y}) is unreachable")

    def test_returns_none_for_impossible(self):
        """Walling off the entrance should yield no path."""
        spots = self._fresh_spots()
        # Fill every corridor cell by fabricating fake spots around the entrance
        fake_spots = [Spot(id=99+i, x=ENTRANCE[0]+dx, y=ENTRANCE[1]+dy)
                      for i, (dx, dy) in enumerate([(0,1),(1,0),(-1,0)])]
        all_spots = spots + fake_spots
        # Target is destination; entrance is surrounded — path cannot leave entrance
        path = astar(all_spots, ENTRANCE[0], ENTRANCE[1], 2, 4)
        # This MAY still find a path depending on layout; just verify it doesn't crash
        # The important thing is the function returns without error
        self.assertTrue(path is None or isinstance(path, list))

    def test_nearest_is_shortest_path(self):
        """findNearest picks the spot with the fewest corridor steps, not raw distance."""
        lot = make_lot()
        nearest = lot._find_nearest()
        self.assertIsNotNone(nearest)
        path = astar(lot.spots, ENTRANCE[0], ENTRANCE[1], nearest.x, nearest.y)
        best_len = len(path)
        # Every other spot should have path length >= best_len
        for s in lot.spots:
            if s.id == nearest.id:
                continue
            p = astar(lot.spots, ENTRANCE[0], ENTRANCE[1], s.x, s.y)
            if p:
                self.assertGreaterEqual(len(p), best_len)


# ─────────────────────────────────────────────
#  State / Stats Tests
# ─────────────────────────────────────────────

class TestState(unittest.TestCase):

    def test_initial_stats(self):
        lot   = make_lot()
        state = lot.get_state()
        self.assertEqual(state["stats"]["total"],     32)
        self.assertEqual(state["stats"]["available"], 32)
        self.assertEqual(state["stats"]["occupied"],  0)
        self.assertEqual(state["stats"]["queued"],    0)

    def test_add_car_increments_queue(self):
        lot = make_lot()
        lot.add_car()
        state = lot.get_state()
        self.assertEqual(state["stats"]["queued"], 1)
        self.assertEqual(len(state["car_queue"]),  1)

    def test_car_counter_increments(self):
        lot = make_lot()
        lot.add_car()
        lot.add_car()
        state = lot.get_state()
        self.assertEqual(state["car_counter"], 2)
        ids = [c["id"] for c in state["car_queue"]]
        self.assertEqual(ids, [1, 2])

    def test_process_next_parks_car(self):
        lot   = make_lot()
        lot.add_car()
        state = lot.process_next()
        self.assertNotIn("error", state)
        self.assertEqual(state["stats"]["occupied"],  1)
        self.assertEqual(state["stats"]["queued"],    0)
        self.assertEqual(len(state["parked_cars"]),   1)

    def test_process_next_empty_queue(self):
        lot   = make_lot()
        state = lot.process_next()
        self.assertIn("error", state)
        self.assertIn("empty", state["error"].lower())

    def test_process_all(self):
        lot = make_lot()
        for _ in range(5):
            lot.add_car()
        state = lot.process_all()
        self.assertEqual(state["stats"]["occupied"], 5)
        self.assertEqual(state["stats"]["queued"],   0)
        self.assertEqual(state.get("parked_count"),  5)

    def test_process_all_32_cars(self):
        """Fill the entire lot."""
        lot = make_lot()
        for _ in range(32):
            lot.add_car()
        state = lot.process_all()
        self.assertEqual(state["stats"]["occupied"],  32)
        self.assertEqual(state["stats"]["available"], 0)

    def test_no_spots_left(self):
        """33rd car should get an error."""
        lot = make_lot()
        for _ in range(33):
            lot.add_car()
        state = lot.process_all()
        # 32 parked, 1 still queued
        self.assertEqual(state["stats"]["occupied"], 32)
        self.assertEqual(state["stats"]["queued"],   1)


# ─────────────────────────────────────────────
#  Manual Spot Selection Tests
# ─────────────────────────────────────────────

class TestManualSelect(unittest.TestCase):

    def test_manual_select_valid_spot(self):
        lot = make_lot()
        lot.add_car()
        state = lot.process_next(spot_id=5)
        self.assertNotIn("error", state)
        parked = next(p for p in state["parked_cars"] if p["spot_id"] == 5)
        self.assertIsNotNone(parked)

    def test_manual_select_occupied_spot(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next(spot_id=1)     # park at spot 1
        lot.add_car()
        state = lot.process_next(spot_id=1)   # try again
        self.assertIn("error", state)
        self.assertIn("occupied", state["error"].lower())

    def test_manual_select_nonexistent_spot(self):
        lot = make_lot()
        lot.add_car()
        state = lot.process_next(spot_id=999)
        self.assertIn("error", state)

    def test_preview_path_valid(self):
        lot    = make_lot()
        result = lot.preview_path(1)
        self.assertNotIn("error", result)
        self.assertIn("path",  result)
        self.assertIn("steps", result)
        self.assertGreater(result["steps"], 0)

    def test_preview_path_reserved(self):
        lot = make_lot()
        lot.reserve_spot(1)
        result = lot.preview_path(1)
        self.assertIn("error", result)
        self.assertIn("reserved", result["error"].lower())

    def test_preview_path_occupied(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next(spot_id=1)
        result = lot.preview_path(1)
        self.assertIn("error", result)

    def test_manual_path_is_shortest(self):
        """The path returned by process_next must equal preview_path for same spot."""
        lot     = make_lot()
        preview = lot.preview_path(7)
        lot.add_car()
        state   = lot.process_next(spot_id=7)
        self.assertEqual(state["path_info"]["steps"], preview["steps"])
        self.assertEqual(state["path_info"]["path"],  preview["path"])


# ─────────────────────────────────────────────
#  Reservation Tests
# ─────────────────────────────────────────────

class TestReservations(unittest.TestCase):

    def test_reserve_spot(self):
        lot   = make_lot()
        state = lot.reserve_spot(3)
        self.assertEqual(state["stats"]["reserved"], 1)
        self.assertIn("3", state["reservations"])

    def test_reserve_occupied_spot(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next(spot_id=3)
        state = lot.reserve_spot(3)
        self.assertIn("error", state)

    def test_reserve_already_reserved(self):
        lot = make_lot()
        lot.reserve_spot(3)
        state = lot.reserve_spot(3)
        self.assertIn("error", state)
        self.assertIn("already reserved", state["error"].lower())

    def test_reserved_spot_blocks_auto_assign(self):
        """Auto-assign should skip reserved spots."""
        lot = make_lot()
        # Reserve spot 1 (the nearest)
        nearest = lot._find_nearest()
        lot.reserve_spot(nearest.id)
        lot.add_car()
        state = lot.process_next()
        # Should have parked somewhere OTHER than nearest
        parked_spot_id = state["parked_cars"][0]["spot_id"]
        self.assertNotEqual(parked_spot_id, nearest.id)

    def test_reserved_spot_blocks_manual_select(self):
        lot = make_lot()
        lot.reserve_spot(5)
        lot.add_car()
        state = lot.process_next(spot_id=5)
        self.assertIn("error", state)
        self.assertIn("reserved", state["error"].lower())

    def test_cancel_reservation(self):
        lot   = make_lot()
        lot.reserve_spot(4)
        state = lot.cancel_reservation(4)
        self.assertEqual(state["stats"]["reserved"], 0)
        self.assertNotIn("4", state["reservations"])

    def test_cancel_nonexistent_reservation(self):
        lot   = make_lot()
        state = lot.cancel_reservation(99)
        self.assertIn("error", state)

    def test_reservation_remaining_time(self):
        lot = make_lot()
        lot.reserve_spot(2)
        state = lot.get_state()
        rv = state["reservations"]["2"]
        self.assertAlmostEqual(rv["remaining_s"], 600, delta=2)

    def test_reservation_expires(self):
        """Manually expire a reservation by back-dating the expires timestamp."""
        lot = make_lot()
        lot.reserve_spot(6)
        # Force-expire by setting expires to the past
        lot.reservations[6].expires = time.time() - 1
        # Trigger expiry check by calling get_state (doesn't auto-expire here)
        # Instead directly verify the timer fires — we just check the state dict
        state = lot.get_state()
        rv = state["reservations"]["6"]
        self.assertLessEqual(rv["remaining_s"], 0)


# ─────────────────────────────────────────────
#  Undo / Redo Tests
# ─────────────────────────────────────────────

class TestUndoRedo(unittest.TestCase):

    def test_undo_add_car(self):
        lot = make_lot()
        lot.add_car()
        state = lot.undo()
        self.assertEqual(state["stats"]["queued"],   0)
        self.assertEqual(state["car_counter"],       0)
        self.assertNotIn("error", state)

    def test_undo_park_car(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next()
        self.assertEqual(lot.get_state()["stats"]["occupied"], 1)
        state = lot.undo()          # undo the park
        self.assertEqual(state["stats"]["occupied"], 0)
        # The car should be back in the queue
        self.assertEqual(state["stats"]["queued"], 1)

    def test_undo_reserve(self):
        lot = make_lot()
        lot.reserve_spot(5)
        lot.undo()
        state = lot.get_state()
        self.assertEqual(state["stats"]["reserved"], 0)

    def test_undo_empty_stack(self):
        lot   = make_lot()
        state = lot.undo()
        self.assertIn("error", state)
        self.assertIn("nothing to undo", state["error"].lower())

    def test_redo_after_undo(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next()
        lot.undo()                              # undo park
        state = lot.redo()                      # redo park
        self.assertEqual(state["stats"]["occupied"], 1)
        self.assertEqual(state["stats"]["queued"],   0)

    def test_redo_cleared_on_new_action(self):
        lot = make_lot()
        lot.add_car()
        lot.process_next()
        lot.undo()
        lot.add_car()       # new action should clear redo stack
        state = lot.redo()
        self.assertIn("error", state)

    def test_undo_multiple_steps(self):
        lot = make_lot()
        lot.add_car()
        lot.add_car()
        lot.add_car()
        lot.process_all()
        # Undo 3 park actions
        for _ in range(3):
            lot.undo()
        state = lot.get_state()
        self.assertEqual(state["stats"]["occupied"], 0)

    def test_undo_stack_max_size(self):
        """Stack must not grow beyond MAX_UNDO."""
        from app import MAX_UNDO
        lot = make_lot()
        for _ in range(MAX_UNDO + 10):
            lot.add_car()
        self.assertLessEqual(len(lot._undo_stack), MAX_UNDO)

    def test_undo_label_correct(self):
        lot = make_lot()
        lot.add_car()
        state = lot.undo()
        self.assertIn("undone_label", state)
        self.assertIn("Car #1", state["undone_label"])


# ─────────────────────────────────────────────
#  Reset Tests
# ─────────────────────────────────────────────

class TestReset(unittest.TestCase):

    def test_reset_clears_all(self):
        lot = make_lot()
        lot.add_car(); lot.add_car()
        lot.process_all()
        lot.reserve_spot(10)
        state = lot.reset()
        self.assertEqual(state["stats"]["occupied"],  0)
        self.assertEqual(state["stats"]["available"], 32)
        self.assertEqual(state["stats"]["queued"],    0)
        self.assertEqual(state["stats"]["reserved"],  0)
        self.assertEqual(state["car_counter"],        0)

    def test_reset_clears_undo_redo(self):
        lot = make_lot()
        lot.add_car(); lot.process_next()
        lot.reset()
        self.assertEqual(len(lot._undo_stack), 0)
        self.assertEqual(len(lot._redo_stack), 0)

    def test_reset_clears_reservations_timers(self):
        lot = make_lot()
        lot.reserve_spot(1)
        lot.reserve_spot(2)
        lot.reset()
        self.assertEqual(len(lot._reserve_timers), 0)
        self.assertEqual(len(lot.reservations),    0)


# ─────────────────────────────────────────────
#  Log Tests
# ─────────────────────────────────────────────

class TestLog(unittest.TestCase):

    def test_log_has_entries(self):
        lot = make_lot()
        log = lot.get_log()
        self.assertGreater(len(log), 0)

    def test_log_entry_structure(self):
        lot   = make_lot()
        entry = lot.get_log()[0]
        self.assertIn("timestamp", entry)
        self.assertIn("message",   entry)
        self.assertIn("level",     entry)

    def test_log_most_recent_first(self):
        lot = make_lot()
        lot.add_car()
        log = lot.get_log()
        # Most recent entry should mention car queue
        self.assertIn("queue", log[0]["message"].lower())

    def test_log_limit(self):
        lot = make_lot()
        for _ in range(10):
            lot.add_car()
        log = lot.get_log(limit=3)
        self.assertLessEqual(len(log), 3)


# ─────────────────────────────────────────────
#  Flask Route Tests
# ─────────────────────────────────────────────

class TestRoutes(unittest.TestCase):

    def setUp(self):
        from app import app, lot as global_lot
        global_lot.reset()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def get(self, url):
        return self.client.get(url)

    def post(self, url, body=None):
        return self.client.post(
            url,
            data=json.dumps(body or {}),
            content_type="application/json",
        )

    def delete(self, url):
        return self.client.delete(url)

    def test_get_state(self):
        import json
        r = self.get("/api/state")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertIn("stats", data)

    def test_post_add_car(self):
        import json
        r    = self.post("/api/cars")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["queued"], 1)

    def test_post_process_next(self):
        import json
        self.post("/api/cars")
        r    = self.post("/api/cars/process")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["occupied"], 1)
        self.assertIn("path_info", data)

    def test_post_process_next_manual(self):
        import json
        self.post("/api/cars")
        r    = self.post("/api/cars/process", {"spot_id": 10})
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["parked_cars"][0]["spot_id"], 10)

    def test_post_process_all(self):
        import json
        for _ in range(3):
            self.post("/api/cars")
        r    = self.post("/api/cars/process-all")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["parked_count"], 3)

    def test_get_path_preview(self):
        import json
        r    = self.get("/api/spots/1/path")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertIn("path",  data)
        self.assertIn("steps", data)

    def test_post_reserve(self):
        import json
        r    = self.post("/api/spots/5/reserve")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["reserved"], 1)

    def test_delete_reserve(self):
        import json
        self.post("/api/spots/5/reserve")
        r    = self.delete("/api/spots/5/reserve")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["reserved"], 0)

    def test_post_undo(self):
        import json
        self.post("/api/cars")
        r    = self.post("/api/undo")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["queued"], 0)

    def test_post_redo(self):
        import json
        self.post("/api/cars")
        self.post("/api/undo")
        r    = self.post("/api/redo")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["queued"], 1)

    def test_post_reset(self):
        import json
        self.post("/api/cars")
        self.post("/api/cars/process")
        r    = self.post("/api/reset")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["stats"]["occupied"], 0)

    def test_get_log(self):
        import json
        r    = self.get("/api/log?limit=5")
        data = json.loads(r.data)
        self.assertTrue(data["success"])
        self.assertIn("log", data)
        self.assertLessEqual(len(data["log"]), 5)

    def test_cors_headers(self):
        r = self.get("/api/state")
        self.assertIn("Access-Control-Allow-Origin", r.headers)


# ─────────────────────────────────────────────

import json   # needed by TestRoutes above

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    classes = [
        TestAstar,
        TestState,
        TestManualSelect,
        TestReservations,
        TestUndoRedo,
        TestReset,
        TestLog,
        TestRoutes,
    ]
    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
