"""Car and sensor faults: a healthy car is unchanged, each fault does what it says, and the
episode loop can be stepped and handed between a person and the controller."""

import numpy as np

from ugv.park.episode import Episode, Scenario, run
from ugv.park.faults import FAMILIES, Faults, FaultState, J
from ugv.park.geometry import bay, compose, inverse


def _state(**kw):
    return FaultState(Faults(**kw), np.random.default_rng(0))


def test_no_fault_is_the_identity():
    fs = _state()
    v = np.array([0.3, -0.1, 0.5])
    assert np.allclose(fs.wheels(v, 0.02), v)
    assert np.allclose(fs.command(np.array([0.2, 0, 0]), 0.02), [0.2, 0, 0])


def test_a_weak_wheel_turns_a_straight_drive():
    v = _state(wheel_gain=(0.6, 1, 1, 1)).wheels(np.array([0.3, 0, 0]), 0.02)
    assert abs(v[2]) > 0.05  # yaw appears from a forward command


def test_delay_and_battery():
    fs = _state(delay_ticks=2, battery=0.8)
    out = [fs.command(np.array([0.3, 0, 0]), 0.02)[0] for _ in range(4)]
    assert out[:2] == [0, 0] and abs(out[2] - 0.24) < 1e-9


def test_sample_respects_only_and_severity():
    rng = np.random.default_rng(1)
    for fam in FAMILIES:
        assert Faults.sample(rng, 1.0, only=fam).describe(), fam
    assert Faults.sample(rng, 0.0) == Faults()
    assert J.shape == (4, 3)


def test_close_starts_are_close_and_clear():
    for s in range(30):
        sc = Scenario.sample(np.random.default_rng(s), start="close")
        rel = compose(inverse(np.array(sc.airframe)), np.array(sc.car))
        assert 0.39 < np.hypot(*rel[:2]) < 1.31
        assert not bay().collides(rel)


def test_episode_loop_matches_run_and_faults_change_the_outcome():
    sc = Scenario.sample(np.random.default_rng(4))
    ep = Episode(sc, seed=4)
    while not ep.tick():
        pass
    assert ep.result().time_s == run(sc, seed=4).time_s
    faulty = Scenario(sc.airframe, sc.car, sc.plant_scale, faults=1.0, only="wheel")
    assert run(faulty, seed=4).faults


def test_manual_ticks_do_not_need_a_controller():
    ep = Episode(Scenario((0, 0, 0), (-1.4, 0.0, 0.0), 0.0), seed=0, engaged=False)
    for _ in range(50):
        ep.tick(np.array([0.3, 0, 0]))
    assert ep.state == "MANUAL" and ep.true_rel()[0] > -1.4 + 0.1


def test_pitch_refit_recovers_a_knocked_camera_from_five_points():
    from ugv.park.perception import (
        CameraError,
        DetectorModel,
        _model_points,
        _true_projection,
        camera_points,
        solve,
    )

    rel = np.array([0.6, 0.02, 0.05])  # airframe 0.6 m ahead, near the bay
    knocked = CameraError(pitch=np.radians(-4))
    uv, _ = _true_projection(camera_points(rel), knocked)
    _, model = _model_points()
    idx = [0, 4, 5, 9, 10]  # five points: pitch is not freed by the normal path
    prior = np.array([*rel, 0.0])
    det = DetectorModel(pixel_std=0.0)
    m = solve(model[idx], uv[idx], det, prior)
    assert m is not None and len(m) > 5 and m[5] == 1  # re-fitted
    assert abs(m[3] - np.radians(4)) < np.radians(1)  # pitch found (sign: positive = up)
    assert abs(m[0] - rel[0]) < 0.02
    old = solve(model[idx], uv[idx], DetectorModel(pixel_std=0.0, refit_pitch=False), prior)
    assert old is None or abs(old[0] - rel[0]) > abs(m[0] - rel[0])  # without it: lost or worse


def test_a_flatlined_gyro_is_replaced_by_the_motor_model():
    from ugv.park.estimator import Estimator

    est = Estimator(gyro_check=True)
    for k in range(30):
        est.predict(k * 0.02, np.array([0.0, 0.0, 0.2]), 0.0, 0.02)  # turning, gyro reads 0
    assert est.gyro_dead and est.odom[2] > 0.0  # heading still advances, from the model


def test_a_dead_camera_ends_in_a_safe_abort():
    from dataclasses import replace

    from ugv.park.controller import ParkParams

    sc = Scenario(
        (0, 0, 0), (-1.4, 0.0, 0.0), 1.0, overrides=(("broken", "camera"), ("broken_at_s", 0.0))
    )
    params = replace(ParkParams(), camera_silent_s=3.0)
    r = run(sc, params, seed=3)
    assert r.outcome == "abort:camera silent" and r.time_s < 5


def test_environment_overrides_reach_the_faults():
    sc = Scenario((0, 0, 0), (-1.4, 0.0, 0.0), 1.0, overrides=(("phantom_rate", 0.04),))
    ep = Episode(sc, seed=1)
    assert ep.faults is not None and ep.faults.f.phantom_rate == 0.04
