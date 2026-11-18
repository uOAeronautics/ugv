"""Parking simulator: bay geometry from the mesh, floor-constrained pose, motor ID, one episode."""

from __future__ import annotations

import numpy as np

from ugv.park.actuator import Axis
from ugv.park.controller import TABLE
from ugv.park.episode import Episode, Scenario, run
from ugv.park.geometry import bay, compose, inverse
from ugv.park.perception import DetectorModel, _model_points, project_planar, solve, solve_planar
from ugv.park.plant import Floor, Plant, PlantParams


def test_bay_is_a_tail_entry_between_the_gears() -> None:
    b = bay()
    lo, hi = b.free_x
    assert lo < b.target[0] < hi, "target lies between main-gear axle and nose wheel"
    assert b.target[2] == 0.0 and b.clearance_m > 0.02
    assert not b.collides(b.target)
    assert b.collides(np.array([lo, 0.06, 0.0])), "between the main wheels, 6 cm off is a hit"


def test_compose_inverse_round_trip() -> None:
    a, c = np.array([0.3, -1.2, 2.0]), np.array([-0.5, 0.4, -1.0])
    assert np.allclose(compose(inverse(a), compose(a, c)), c)


def test_clean_floor_moves_exactly_as_commanded() -> None:
    p = Plant(PlantParams(), np.zeros(3), np.random.default_rng(0))
    p.step(np.array([0.3, 0.0, 0.0]), 0.02)
    assert np.allclose(p.ground_v, p.vel[:2], atol=1e-12)


def test_slippery_floor_slides_sideways_in_a_turn() -> None:
    p = Plant(
        PlantParams(),
        np.zeros(3),
        np.random.default_rng(0),
        floor=Floor(grip=0.7, slip=1.0, patch=0.0),
    )
    for _ in range(30):  # 0.6 s of forward and yaw: the car cannot hold the corner
        _, gyro = p.step(np.array([0.2, 0.0, 0.4]), 0.02)
    assert abs(p.pose[2]) > 0.5, "it still turns, just less than commanded"
    assert abs(gyro) > 0.5
    body = [
        np.cos(p.pose[2]) * p.ground_v[0] + np.sin(p.pose[2]) * p.ground_v[1],
        -np.sin(p.pose[2]) * p.ground_v[0] + np.cos(p.pose[2]) * p.ground_v[1],
    ]
    assert abs(body[1]) > 0.01, "the ground velocity lags the heading: it skids"


def test_floor_grip_changes_from_patch_to_patch() -> None:
    f = Floor(grip=0.95, slip=0.2, patch=0.5, phase=(0.3, 1.1))
    grips = [f.grip_at(x, y) for x in np.linspace(-1.8, 1.8, 7) for y in np.linspace(-1.8, 1.8, 7)]
    assert max(grips) - min(grips) > 0.15


def test_planar_solve_recovers_pose_and_pitch() -> None:
    _, model = _model_points()
    truth = np.array([1.2, 0.1, 2.5, np.radians(1.5)])
    obs = project_planar(model, truth)
    q, res = solve_planar(model, obs, np.array([1.0, 0.0, 2.3, 0.0]))
    assert np.allclose(q, truth, atol=1e-4) and res.max() < 1e-3


def test_two_point_solve_keeps_prior_yaw() -> None:
    _, model = _model_points()
    truth = np.array([0.5, 0.01, 0.0, 0.0])
    pts = model[[0, 9]]  # nose and nose wheel: both on the centreline
    m = solve(pts, project_planar(pts, truth), DetectorModel(), np.array([0.45, 0.0, 0.0, 0.0]))
    assert m is not None and abs(m[0] - 0.5) < 1e-3 and abs(m[1] - 0.01) < 1e-3


def test_axis_identifies_gain_and_deadzone() -> None:
    ax = Axis(2.3, 0.05, 0.01)
    for d in (0.1, 0.15, 0.2, 0.25, 0.3) * 4:
        ax.observe(d, 1.8 * (d - 0.09))
    assert abs(ax.gain - 1.8) < 0.15 and abs(ax.deadzone - 0.09) < 0.02
    ax.observe(0.08, 0.0)
    assert ax.deadzone >= 0.085, "a stall raises the deadzone"


def test_every_state_is_declared() -> None:
    states = set(TABLE) | set().union(*TABLE.values())
    assert states == set(TABLE)


def test_lost_frame_is_reported_without_the_previous_faults() -> None:
    sc = Scenario(
        (0.0, 0.0, 0.0),
        (1.2, 0.0, 0.0),
        faults=1.0,
        overrides=(("drop_rate", 1.0),),
    )
    ep = Episode(sc, seed=5, engaged=False)
    ep.tick(np.zeros(3))
    assert ep.cam_fx == {"lost": True}
    assert ep.last_kp == []


def test_ideal_episode_parks() -> None:
    det = DetectorModel(
        pixel_std=0.3,
        miss_rate=0,
        outlier_rate=0,
        focal_rel_std=0,
        mount_yaw_std=0,
        mount_pitch_std=0,
    )
    sc = Scenario((0.2, -0.3, 1.0), (1.8, 1.0, -2.0), plant_scale=0.0, detector=det)
    r = run(sc, seed=3)
    assert r.outcome == "parked", r


def test_lab_api_serves_every_view() -> None:
    from fastapi.testclient import TestClient

    from ugv.lab.server import create_app

    c = TestClient(create_app())
    for path in (
        "/",
        "/api/ledger",
        "/api/park/evals",
        "/api/park/searches",
        "/api/park/world",
        "/api/pose",
    ):
        assert c.get(path).status_code == 200, path
    assert c.get("/runs/../pyproject.toml").status_code == 404
    assert c.get("/static/hdri.json").status_code == 200
    assert c.get("/api/park/hdri/not%20an%20id.hdr").status_code == 404
    world = c.get("/api/park/world").json()
    assert len(world["bay"]["obstacles"]) > 10 and world["bay"]["target"][2] == 0.0
    s = c.post(
        "/api/park/session",
        json={"seed": 2, "quality": "healthy", "env": "hdri:industrial_workshop_foundry"},
    )
    assert s.status_code == 200 and len(s.json()["floor"]) == 3


def test_mutation_follows_failure_levers() -> None:
    from collections import Counter

    from ugv.park.controller import ParkParams
    from ugv.park.search import LEVERS, SPACE, mutate

    rng = np.random.default_rng(0)
    hits = Counter()
    for _ in range(200):
        p, change = mutate(ParkParams(), Counter({"collision": 10}), rng)
        assert len(change) == 3
        for k, v in change.items():
            lo, hi = SPACE[k]
            assert lo <= v <= hi and getattr(p, k) == v
            hits[k in LEVERS["collision"]] += 1
    assert hits[True] > hits[False], "failure levers are favoured"
