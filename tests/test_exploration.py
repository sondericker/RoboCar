import math
import random
import tempfile
import unittest
from pathlib import Path

from robocar.exploration import (
    CoverageGrid,
    ExplorationLearner,
    Frontier,
    load_learner,
    save_learner,
)
from explore_sim import CORRIDOR, ROOM_A, ROOM_B, FloorPlan, explore
from robocar.surfaces import SONAR_BEAM_WIDTH_DEGREES, SONAR_FORWARD_OFFSET_CM


def sweep_into(grid: CoverageGrid, floor: FloorPlan, pose) -> None:
    grid.integrate(floor.scan(pose), pose, 25.0)


class GridTests(unittest.TestCase):
    def test_a_long_reading_seen_off_to_one_side_becomes_a_lead(self):
        # From room B the corridor's entrance shows as readings running far
        # deeper than the wall either side of it.
        floor, grid = FloorPlan(), CoverageGrid()
        sweep_into(grid, floor, (-520.0, 100.0, -90.0))
        self.assertEqual(len(grid.leads), 1)
        mouth, into = grid.leads[0]
        self.assertAlmostEqual(mouth[0], -400.0, delta=20.0)
        self.assertTrue(40.0 < mouth[1] < 120.0)
        self.assertGreater(into[0], 0.9)  # pointing into the corridor

    def test_missing_echoes_do_not_clear_space_through_corridor_walls(self):
        # Sound glances off a corridor's side walls and never comes back; the
        # map must not take that as open space beyond them.
        floor, grid = FloorPlan(), CoverageGrid()
        for x in (-200.0, -250.0, -300.0):
            for heading in (90.0, 75.0, 105.0):
                sweep_into(grid, floor, (x, 80.0, heading))
        free = grid.free()
        for x in range(-330, -170, 10):
            for y in (20.0, 140.0):  # 20 cm outside each wall
                cell = grid.cell_of((x, y))
                self.assertFalse(free[cell], (x, y))

    def test_a_flat_wall_close_ahead_is_mapped(self):
        floor, grid = FloorPlan(), CoverageGrid()
        sweep_into(grid, floor, (99.0, -7.0, -90.0))  # facing room A's right wall, 51 cm away
        occupied = grid.occupied()
        marked = sum(occupied[grid.cell_of((150.0 + dx, y))] for y in range(-30, 20, 5) for dx in (-2.5, 2.5))
        self.assertGreaterEqual(marked, 6)

    def test_a_corridor_is_recognized_with_its_centre_line(self):
        floor, grid = FloorPlan(), CoverageGrid()
        for x in (-180.0, -220.0, -260.0, -300.0):
            for heading in (90.0, 60.0, 120.0):
                sweep_into(grid, floor, (x, 85.0, heading))
        corridor = grid.corridor((-250.0, 70.0, 100.0))
        self.assertIsNotNone(corridor)
        self.assertAlmostEqual(corridor.axis_deg, 90.0, delta=5.0)
        # The centre line (y = 80) is 10 cm to the car's right facing west.
        self.assertAlmostEqual(corridor.centre_offset_cm, -10.0, delta=4.0)
        self.assertAlmostEqual(corridor.width_cm, 75.0, delta=10.0)
        self.assertIsNone(grid.corridor((0.0, 0.0, 0.0)))

    def test_inside_the_wall_margin_the_car_never_drives_closer_to_the_wall(self):
        floor, grid = FloorPlan(), CoverageGrid()
        for heading in (90.0, 60.0, 120.0, 180.0, 0.0):
            sweep_into(grid, floor, (-250.0, 80.0, heading))
        # 15 cm from the corridor's lower wall: angled toward it, no room;
        # angled away from it, free to go.
        self.assertLess(grid.clear_run((-250.0, 55.0, 110.0), 60.0), 5.0)
        self.assertGreater(grid.clear_run((-250.0, 55.0, 70.0), 60.0), 30.0)


class ScanMatchTests(unittest.TestCase):
    """Fitting a sweep to the coverage map when walls won't register."""

    @classmethod
    def setUpClass(cls):
        from hunt_sim import build_map  # the house as a mapping run leaves it

        cls.floor = FloorPlan()
        cls.grid = build_map(cls.floor)

    def test_a_trusted_match_puts_the_car_roughly_where_it_is(self):
        rng = random.Random(5)
        trusted = 0
        cases = 0
        while cases < 30:
            truth = (rng.uniform(-120, 120), rng.uniform(-70, 120), rng.uniform(0, 360))
            if self.floor.clearance(truth[:2]) < 25:
                continue
            cases += 1
            guess = (truth[0] + rng.uniform(-20, 20), truth[1] + rng.uniform(-20, 20), truth[2] + rng.uniform(-15, 15))
            match = self.grid.match_scan(self.floor.scan(truth), guess, 24.0, 20.0)
            self.assertIsNotNone(match)
            self.assertGreaterEqual(match.score, match.prior_score - 0.05)
            if match.good:
                trusted += 1
                self.assertLess(math.hypot(match.pose[0] - truth[0], match.pose[1] - truth[1]), 20.0, match.summary())
                self.assertLess(abs((match.pose[2] - truth[2] + 180) % 360 - 180), 8.0, match.summary())
        self.assertGreaterEqual(trusted, 5)

    def test_a_sweep_of_one_long_wall_is_not_trusted(self):
        # A single wall pins how far away it is, not where along it the car stands.
        grid = CoverageGrid()

        def wall_scan(pose):
            readings = []
            for angle in range(-90, 91, 5):
                bearing = math.radians(pose[2] + angle)
                forward = math.cos(bearing)
                origin_y = pose[1] + math.cos(math.radians(pose[2])) * SONAR_FORWARD_OFFSET_CM
                distance = (150.0 - origin_y) / forward if forward > 0.2 else None
                readings.append({"angle_deg": float(angle), "distance_cm": distance if distance and distance < 250 else 300.0})
            return readings

        for x in range(-150, 151, 30):
            pose = (float(x), 0.0, 0.0)
            grid.integrate(wall_scan(pose), pose, SONAR_BEAM_WIDTH_DEGREES)
        match = grid.match_scan(wall_scan((0.0, 0.0, 0.0)), (18.0, 0.0, 0.0), 24.0, 10.0)
        self.assertTrue(match is None or not match.good, match and match.summary())

    def test_no_map_no_match(self):
        self.assertIsNone(CoverageGrid().match_scan(self.floor.scan((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0), 20.0, 10.0))


class LearnerTests(unittest.TestCase):
    def test_only_pursuits_that_arrived_teach_anything(self):
        learner = ExplorationLearner()
        before = dict(learner.weights)
        doorway = Frontier((0.0, 100.0), (0.0, 100.0), 10, 100.0, 1000.0, "doorway")
        elsewhere = Frontier((300.0, 0.0), (300.0, 0.0), 10, 300.0, 1000.0, "edge")
        # Given up halfway: no lesson.
        learner.observe((0.0, 0.0, 0.0), 0)
        learner.pursue(doorway, 0)
        learner.observe((0.0, 40.0, 0.0), 300)
        learner.pursue(elsewhere, 300)
        self.assertEqual(learner.weights, before)
        # Arrived at a doorway that showed far more than an ordinary edge would.
        learner.pursue(doorway, 300)
        learner.observe((0.0, 90.0, 0.0), 300)
        learner.observe((0.0, 95.0, 0.0), 1800)
        learner.pursue(elsewhere, 1800)
        self.assertGreater(learner.weights["doorway"], before["doorway"])
        self.assertEqual(learner.pursuits, 1)

    def test_effort_price_follows_what_moves_have_been_buying(self):
        learner = ExplorationLearner()
        learner.observe((0.0, 0.0, 0.0), 0)
        for step in range(1, 6):
            learner.observe((0.0, 50.0 * step, 0.0), 1000 * step)  # 20 cells per cm
        self.assertGreater(learner.rate, 10.0)

    def test_what_was_learned_survives_a_restart(self):
        learner = ExplorationLearner()
        learner.weights["doorway"] = 1.7
        learner.rate = 9.0
        learner.pursuits = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exploration.json"
            save_learner(path, learner)
            again = load_learner(path)
            self.assertEqual(again.weights["doorway"], 1.7)
            self.assertEqual(again.rate, 9.0)
            self.assertEqual(again.pursuits, 4)
            path.write_text("not json")
            self.assertEqual(load_learner(path).weights, ExplorationLearner().weights)


class FloorPlanTests(unittest.TestCase):
    """Two rooms joined by a corridor, with missing echoes reading 300 cm."""

    def explores_both_rooms(self, start, sweeps: int) -> None:
        learner = ExplorationLearner()
        from robocar.mapping import plan_explore_step

        floor, grid, history, bumps = explore(
            sweeps, plan=lambda *args: plan_explore_step(*args, learner=learner), start=start
        )
        self.assertEqual(bumps, 0)
        for box in (ROOM_A, CORRIDOR, ROOM_B):
            self.assertGreater(floor.room_coverage(grid, box), 0.75, box)
        self.assertGreater(floor.coverage(grid), 0.85)

    def test_starting_in_the_first_room_it_finds_the_corridor_and_the_room_beyond(self):
        self.explores_both_rooms((-100.0, -60.0, 45.0), 75)

    def test_starting_in_the_far_room_it_finds_its_way_back(self):
        self.explores_both_rooms((-600.0, 0.0, 0.0), 40)


if __name__ == "__main__":
    unittest.main()
