"""The Lab dashboard's server side: live sessions shared by a person and the controller, named and
placed starts, and the APIs."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ugv.lab import sim
from ugv.lab.server import create_app
from ugv.park.geometry import bay
from ugv.pose.detector import default_runs


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


def test_fixed_high_exposure_defect():
    s = sim.Sessions()
    r = s.create(5, preset="tail", quality="healthy", exposure=2.0)
    assert any("exposure stuck at +2.0" in f for f in r["faults"])
    assert r["camera_sim"]["exposure_stops"] == 2.0  # the browser pins the render to it
    clean = s.create(5, preset="tail", quality="healthy")
    assert clean["camera_sim"]["exposure_stops"] == 0.0
    assert not any("exposure stuck" in f for f in clean["faults"])


def test_external_camera_feeds_the_estimator():
    """True detections, computed off-world, must localize the car — the strict protocol path."""
    from ugv.park.geometry import compose, inverse
    from ugv.park.perception import DetectorModel, model_points, project_planar

    s = sim.Sessions()
    r = s.create(5, preset="nose-on", quality="healthy", env="studio")
    det = DetectorModel()
    q = np.array(r["start"])
    rel = compose(inverse(q[:3]), np.array([0.0, 0.0, 0.0]))
    uv = project_planar(model_points(det), np.array([rel[0], rel[1], rel[2], 0.0]))
    dets = [
        [int(i), float(uv[i, 0]), float(uv[i, 1])]
        for i in range(len(uv))
        if 0 <= uv[i, 0] < 1280 and 0 <= uv[i, 1] < 720
    ]
    assert len(dets) >= 6
    est = None
    for _ in range(6):
        s.camera_frame(r["id"], dets, False, {"exposure": 0.0, "blur": 0.0})
        snap = s.step(r["id"], 10, None, auto=True)
        est = snap["est"]
    assert est is not None


def test_live_exposure_change_no_restart():
    s = sim.Sessions()
    r = s.create(5, preset="tail", quality="healthy", env="studio")  # zero faults drawn
    s.step(r["id"], 10, [0, 0, 0], auto=False)
    s.set_exposure(r["id"], 2.0)  # pin it mid-round
    out = s.camera_frame(r["id"], [[0, 640, 360]], False, {"exposure": 2.0, "blur": 0.0})
    assert out["cam_fx"]["exposure"] == 2.0
    assert out["t"] == pytest.approx(0.2)  # same round kept running, not a new one
    s.step(r["id"], 5, [0, 0, 0], auto=False)
    s.set_exposure(r["id"], 0.0)  # release back to auto
    out = s.camera_frame(r["id"], [[0, 640, 360]], False, {"exposure": 0.0, "blur": 0.0})
    assert out["cam_fx"]["exposure"] == 0.0
    lost = s.camera_frame(r["id"], None, True, None)
    assert lost["kp"] == [] and lost["cam_fx"]["lost"] is True
    with pytest.raises(KeyError):
        s.set_exposure("nope", 1.0)


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
    assert c.post("/api/park/session", json={"seed": 1, "exposure": 5}).status_code == 400
    live = c.post(
        "/api/park/session",
        json={"seed": 1, "preset": "tail", "quality": "healthy", "env": "studio"},
    ).json()
    assert live["camera_sim"]["exposure_stops"] == 0.0
    assert c.post(f"/api/park/session/{live['id']}/exposure", json={"stops": 9}).status_code == 400
    assert c.post("/api/park/session/nope/exposure", json={"stops": 1}).status_code == 404
    assert c.post(f"/api/park/session/{live['id']}/exposure", json={"stops": 2}).status_code == 200
    frame = f"/api/park/session/{live['id']}/frame"
    assert c.post(frame, content=b"\0" * 100).status_code == 400  # not a 1280x720 mono frame
    assert c.post("/api/park/session/nope/frame?lost=true").status_code == 404
    lost = c.post(f"{frame}?lost=true&exposure=2").json()
    assert lost["kp"] == [] and lost["cam_fx"]["lost"]
    r = c.post("/api/park/session", json={"seed": 1, "preset": "nose-on"}).json()
    step = c.post(f"/api/park/session/{r['id']}/step", json={"ticks": 5, "auto": True})
    assert step.status_code == 200 and {"car", "est", "state", "kp"} <= set(step.json())
    assert c.post("/api/park/session/nope/step", json={"ticks": 1}).status_code == 404
    plan = c.get("/api/park/plan").json()
    assert (
        len(plan["skeleton"]["names"]) == 15
        and len(plan["skeleton"]["points"]) == 15
        and len(plan["tris"][0]) == 8
    )
    assert c.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert c.get("/runs/../pyproject.toml").status_code == 404


@pytest.mark.skipif(
    not all((p / "model.onnx").exists() for p in default_runs()),
    reason="trained detector not on this machine (runs/pose_train, not in Git)",
)
def test_frames_go_through_the_trained_detector():
    """A rendered frame is read by the network, never by anything that knows the pose."""
    c = TestClient(create_app())
    live = c.post("/api/park/session", json={"seed": 3, "preset": "tail"}).json()
    out = c.post(
        f"/api/park/session/{live['id']}/frame", content=np.zeros(1280 * 720, np.uint8).tobytes()
    )
    assert out.status_code == 200
    det = out.json()["det"]
    assert det["ms"] > 0 and all(0 <= int(i) < 12 for i, _, _ in det["points"])
