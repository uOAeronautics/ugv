"""The Lab dashboard's server side: live sessions shared by a person and the controller, named and
placed starts, and the APIs."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ugv.lab import sim
from ugv.lab.server import create_app
from ugv.park.geometry import bay


def test_presets_are_clear_of_the_airframe():
    for name, q in sim.PRESETS.items():
        assert not bay().collides(np.array(q), 0.01), name


def test_random_starts_are_clear_and_inside_the_room():
    for seed in range(20):
        q = sim.start_pose(seed, None, None)
        assert max(abs(q[:2])) < 1.8 and not bay().collides(q)


def test_a_start_inside_the_airframe_is_refused():
    with pytest.raises(ValueError):
        sim.start_pose(0, [-0.16, 0.13, 0.0], None)  # on the main gear


def test_hand_over_to_the_controller_and_back():
    s = sim.Sessions()
    r = s.create(5, preset="tail")
    assert r["state"] == "MANUAL" and r["faults"] is not None
    moved = s.step(r["id"], 25, [0.3, 0, 0], auto=False)
    assert moved["car"][0] > r["car"][0]  # driven forward along +x, towards the airframe
    auto = s.step(r["id"], 25, None, auto=True)
    assert auto["state"] != "MANUAL"
    back = s.step(r["id"], 5, [0, 0, 0], auto=False)
    assert back["state"] == "MANUAL"


def test_api():
    c = TestClient(create_app())
    assert c.post("/api/park/session", json={"seed": 1, "preset": "bogus"}).status_code == 400
    r = c.post("/api/park/session", json={"seed": 1, "preset": "nose-on"}).json()
    step = c.post(f"/api/park/session/{r['id']}/step", json={"ticks": 5, "auto": True})
    assert step.status_code == 200 and {"car", "est", "state", "kp"} <= set(step.json())
    assert c.post("/api/park/session/nope/step", json={"ticks": 1}).status_code == 404
    plan = c.get("/api/park/plan").json()
    assert len(plan["skeleton"]["names"]) == 12 and len(plan["tris"][0]) == 8
    assert c.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert c.get("/runs/../pyproject.toml").status_code == 404
