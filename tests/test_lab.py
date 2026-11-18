"""The Lab dashboard's server side: rounds draw the same car as the simulator, the autopilot is the
real episode, perception sees the airframe from behind the tail, and the APIs answer."""

import numpy as np
from fastapi.testclient import TestClient

from ugv.lab import sim
from ugv.lab.server import create_app
from ugv.park.episode import Scenario, run
from ugv.park.geometry import bay


def test_round_draws_the_car_episode_run_draws():
    r = sim.new_round(11, "random")
    res = run(Scenario((0.0, 0.0, 0.0), tuple(r["start"]), 1.0), sim._params(), seed=11)
    again = sim.autopilot(11, "random", r["start"])
    assert again["outcome"] == res.outcome and again["time_s"] == res.time_s
    assert again["trace"] and {"car", "est", "state", "kp"} <= set(again["trace"][0])


def test_round_start_is_clear_and_inside_the_room():
    for seed in range(20):
        q = np.array(sim.new_round(seed, "nominal")["start"])
        assert max(abs(q[:2])) < sim.HALF_ROOM
        assert not bay().collides(q)


def test_nominal_level_has_no_spread():
    r = sim.new_round(3, "nominal")
    assert r["camera_error"] == {"focal_scale": 1.0, "yaw": 0.0, "pitch": 0.0}


def test_perceive_sees_the_airframe_from_behind_the_tail():
    out = sim.perceive(0, "nominal", [-0.9, 0.0, 0.0], tick=0)
    assert len(out["kp"]) >= 6
    assert out["pose"] is not None and abs(out["pose"][0] - 0.9) < 0.1  # airframe ahead of the car
    assert sim.perceive(0, "nominal", [-0.9, 0.0, np.pi], tick=0)["kp"] == []  # facing away


def test_api():
    c = TestClient(create_app())
    assert c.get("/api/park/round", params={"seed": 1, "level": "bogus"}).status_code == 400
    r = c.get("/api/park/round", params={"seed": 1, "level": "nominal"}).json()
    body = {"seed": 1, "level": "nominal", "car": r["start"], "tick": 0}
    assert c.post("/api/park/perceive", json=body).status_code == 200
    plan = c.get("/api/park/plan").json()
    assert len(plan["skeleton"]["names"]) == 12 and len(plan["tris"][0]) == 8
    assert c.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert c.get("/runs/../pyproject.toml").status_code == 404
