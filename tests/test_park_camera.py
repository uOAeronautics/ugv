"""Tests for image-in-the-loop park perception, image-level faults, DAgger dump, and runner flags."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ugv.park.camera import NetworkPerception, apply_image_faults, camera_extrinsics
from ugv.park.controller import ParkParams
from ugv.park.dagger import save_shard, write_dagger
from ugv.park.episode import PRESETS, Episode, Scenario
from ugv.park.evaluate import Conditions, evaluate
from ugv.park.geometry import bay
from ugv.park.perception import (
    HEIGHT,
    WIDTH,
    CameraError,
    DetectorModel,
    F,
    _true_projection,
    camera_points,
)
from ugv.pose.detector import Detection
from ugv.pose.render import Camera


def test_network_perception_raises_when_weights_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UGV_DETECTOR", "missing-whole,missing-crop")
    with pytest.raises(FileNotFoundError, match="missing"):
        NetworkPerception()
    with pytest.raises(FileNotFoundError, match="missing"):
        evaluate(ParkParams(), n=1, cond=Conditions(perception="network"), workers=1)


def test_camera_extrinsics_match_true_projection() -> None:
    rel = np.array([1.1, -0.25, 0.4])
    cam = CameraError(focal_scale=1.03, yaw=0.04, pitch=-0.03, cx_px=12.0, cy_px=-7.0)
    b = bay()
    R, t = camera_extrinsics(rel, cam, b.floor_z)
    cam_model = Camera(
        WIDTH,
        HEIGHT,
        intrinsics=(F * cam.focal_scale, WIDTH / 2 + cam.cx_px, HEIGHT / 2 + cam.cy_px),
    )
    pc = camera_points(rel, DetectorModel(near_field=True))
    pts_model = np.zeros_like(pc)
    from ugv.pose.mesh import keypoints, load_airframe

    pts_model = np.array(list(keypoints(load_airframe(), near=True).values()))
    pc_rt = pts_model @ R.T + t
    uv_rt = (pc_rt @ cam_model.K.T)[:, :2] / pc_rt[:, 2:3]
    uv_ref, _ = _true_projection(pc, cam)
    assert np.allclose(uv_rt, uv_ref, atol=1e-6)


def test_image_faults_modify_pixels() -> None:
    rng = np.random.default_rng(0)
    base = np.full((HEIGHT, WIDTH), 120, dtype=np.uint8)
    base[::4, ::4] = 220
    cache: dict = {}

    occ_img = apply_image_faults(
        base, CameraError(), {"occluder": (10, 20, 110, 120)}, 0.0, rng, cache
    )
    assert np.all(occ_img[25:115, 15:105] == 0)

    over = apply_image_faults(base, CameraError(), {"exposure": 2.0}, 0.0, rng, cache)
    under = apply_image_faults(base, CameraError(), {"exposure": -1.5}, 0.0, rng, cache)
    assert over.mean() > base.mean() and under.mean() < base.mean()

    blurred = apply_image_faults(base, CameraError(), {"blur": 6.0}, 0.0, rng, cache)
    assert blurred.std() < base.std()

    distorted = apply_image_faults(base, CameraError(k1=-0.2), None, 0.0, rng, cache)
    assert not np.array_equal(distorted, base)


def test_episode_routes_stub_network_perception_and_drops_frames() -> None:
    calls = 0

    def stub_detector(gray: np.ndarray) -> Detection:
        nonlocal calls
        calls += 1
        assert gray.shape == (HEIGHT, WIDTH) and gray.dtype == np.uint8
        return Detection([], [0.0] * 15, None, 0.1)

    net = NetworkPerception(detector=stub_detector, keep_overlay=True)
    # Phantom is stripped under image-in-the-loop mode and drop_rate=1.0 suppresses detector calls
    sc_drop = Scenario(
        (0.0, 0.0, 0.0),
        PRESETS["tail"],
        faults=1.0,
        overrides=(("drop_rate", 1.0), ("phantom_rate", 1.0)),
    )
    ep_drop = Episode(sc_drop, seed=7, perception=net)
    assert ep_drop.faults is not None and ep_drop.faults.f.phantom_rate == 0.0
    ep_drop.tick(np.zeros(3))
    assert calls == 0 and ep_drop.cam_fx == {"lost": True}

    # Verify a healthy frame flows through render -> detector -> solve -> estimator
    rel = np.array([1.4, 0.0, 0.0])
    uv_ref, _ = _true_projection(camera_points(rel, DetectorModel(near_field=True)), CameraError())
    pts_true = [
        [i, round(float(uv_ref[i, 0]), 1), round(float(uv_ref[i, 1]), 1)] for i in range(12)
    ]

    def perfect_stub(gray: np.ndarray) -> Detection:
        nonlocal calls
        calls += 1
        return Detection(pts_true, [0.99] * 15, (200.0, 40.0, 500.0), 0.1)

    net2 = NetworkPerception(detector=perfect_stub, dump_frames=True)
    sc = Scenario((0.0, 0.0, 0.0), (-1.4, 0.0, 0.0), plant_scale=0.0)
    ep = Episode(sc, seed=2, perception=net2)
    for _ in range(10):
        ep.tick(np.zeros(3))
    assert calls == 2 and ep.fixes >= 1 and ep.est.car_in_airframe() is not None
    assert net2.ep.frames == 2 and len(net2.ep.samples) == 2


def test_dagger_dump_format_and_eval_cli_flag(tmp_path: Path) -> None:
    samples = []
    for k in range(8):
        samples.append(
            {
                "has_err": int(k % 2 == 0),
                "score": float(k * 10),
                "t": float(k),
                "s1": (
                    np.full((180, 320), k, np.uint8),
                    np.zeros((15, 2), np.float32),
                    np.ones(15, bool),
                    np.zeros(12, np.float32),
                    np.eye(3, dtype=np.float32),
                ),
                "s2": (
                    np.full((256, 256), k, np.uint8),
                    np.zeros((15, 2), np.float32),
                    np.ones(15, bool),
                    np.zeros(12, np.float32),
                    np.eye(3, dtype=np.float32),
                ),
            }
        )
    shards = [save_shard(tmp_path, 0, True, samples)]
    out_dir = tmp_path / "dagger"
    res = write_dagger(out_dir, [s for s in shards if s], dump_max=6)
    assert (
        res["written"] == 6
        and (out_dir / "meta.json").exists()
        and (out_dir / "crop" / "meta.json").exists()
    )
    meta1 = json.loads((out_dir / "meta.json").read_text())
    meta2 = json.loads((out_dir / "crop" / "meta.json").read_text())
    assert meta1["height"] == 180 and meta1["width"] == 320 and not meta1["crop"]
    assert meta2["height"] == 256 and meta2["width"] == 256 and meta2["crop"]
    assert np.load(out_dir / "images.npy").shape == (6, 180, 320)
    assert np.load(out_dir / "crop" / "images.npy").shape == (6, 256, 256)

    # Verify CLI --perception flag & --start mixed work
    r = subprocess.run(
        [
            sys.executable,
            "tools/park/eval.py",
            "--name",
            "smoke_model",
            "--perception",
            "model",
            "--start",
            "mixed",
            "--n",
            "12",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "full " in r.stdout
    data = json.loads((tmp_path / "smoke_model.json").read_text())
    assert set(PRESETS) <= set(data["main"]["by_start"])
    assert "random:ring" in data["main"]["by_start"]
    assert all("end_state" in ep for ep in data["main"]["episodes"])

    # Verify CLI --perception network fails visibly (never falls back to DetectorModel) if weights missing
    import os

    env = {**os.environ, "UGV_DETECTOR": "missing-whole,missing-crop"}
    r_miss = subprocess.run(
        [
            sys.executable,
            "tools/park/eval.py",
            "--name",
            "fail_missing",
            "--perception",
            "network",
            "--n",
            "1",
            "--workers",
            "1",
            "--out",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r_miss.returncode != 0 and "missing" in r_miss.stderr
