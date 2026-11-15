import json
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.observatory.replay as replay_mod
import tools.observatory.server as server_mod
from tools.observatory.data import CALIB_RUNS, parse_log, parse_metrics_jsonl
from tools.observatory.replay import eval_bundle, l7_bundle
from tools.observatory.server import app

client = TestClient(app)

# These integration tests read the gitignored local L7 telemetry (*.jsonl);
# CI only has the tracked meta/summary manifests, so skip there instead of
# faking field data.
_HAS_L7_DATA = CALIB_RUNS.is_dir() and any(
    (d / "ticks.jsonl").exists() for d in CALIB_RUNS.iterdir() if d.is_dir()
)
requires_l7_data = pytest.mark.skipif(
    not _HAS_L7_DATA, reason="local L7 telemetry (calibration/runs/l7/*/ticks.jsonl) not present"
)


def test_overview():
    r = client.get("/api/overview")
    assert r.status_code == 200
    assert "counts" in r.json()


def test_leaderboard():
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    assert "entries" in r.json()


def test_curves_runs():
    r = client.get("/api/curves/runs")
    assert r.status_code == 200


def test_replay_list():
    r = client.get("/api/replay/list")
    assert r.status_code == 200


def test_rollout_validation():
    r = client.post(
        "/api/rollout", json={"policy": "brake", "level": "L0", "episodes": 100, "seed": 1}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_contract_runs():
    r = client.get("/api/contract/runs")
    assert r.status_code == 200


def test_curves():
    runs = client.get("/api/curves/runs").json().get("runs", [])
    ids = [r["run_id"] for r in runs[:2]]
    r = client.get("/api/curves", params={"runs": ",".join(ids)})
    assert r.status_code == 200
    assert "series" in r.json()


@requires_l7_data
def test_replay_bundle_l7():
    bundles = client.get("/api/replay/list").json().get("bundles", [])
    l7 = [b for b in bundles if b["bundle_id"].startswith("l7:")]
    assert l7, "expected at least one l7 bundle in calibration/runs/l7"
    r = client.get(f"/api/replay/bundle/{l7[0]['bundle_id']}")
    assert r.status_code == 200
    d = r.json()
    assert d.get("format") == "ugv_replay_v1"
    assert "episodes" in d and d["episodes"], "l7 bundle must carry episodes"
    ep = d["episodes"][0]
    assert "frames" in ep and "duration" in ep and "steps" in ep
    assert "poses" in d, "legacy poses key must be preserved"
    assert "geometry" in d


@requires_l7_data
def test_contract():
    runs = client.get("/api/contract/runs").json().get("runs", [])
    assert runs, "expected at least one contract run in calibration/runs/l7"
    r = client.get("/api/contract", params={"run_id": runs[0]})
    assert r.status_code == 200
    d = r.json()
    assert "status" in d
    assert isinstance(d.get("checks"), list)
    assert isinstance(d.get("channels"), list)
    assert d["channels"], "contract must surface channel comparisons"
    names = {c["name"] for c in d["channels"]}
    assert {
        "vision_hz",
        "frame_age_p50_ms",
        "dropout_rate",
        "imu_w_std",
        "loop_ms_p50",
        "loop_ms_p95",
    } <= names


def test_path_traversal():
    r = client.get("/api/replay/bundle/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404)
    r = client.get("/api/l7/../end.jpg")
    assert r.status_code in (400, 404)


# ---- Regression tests for the Adversarial Observatory Audit fixes ----


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_parse_metrics_jsonl_short_keys_populate_both(tmp_path):
    p = _write_jsonl(
        tmp_path / "run.metrics.jsonl",
        [
            {
                "iter": 1,
                "succ": 0.5,
                "coll": 0.1,
                "timeout": 0.2,
                "false_park": 0.05,
                "pol": 1.1,
                "val": 2.2,
                "aux": 0.3,
                "dyn": 0.4,
                "std": 0.55,
                "sps": 99,
                "wall_s": 12.3,
                "dr_scale": 0.7,
                "anchor_frac": 0.15,
                "return": 3.3,
            }
        ],
    )
    (pt,) = parse_metrics_jsonl(p)
    assert pt["succ"] == pt["succ_rate"] == 0.5
    assert pt["coll"] == pt["coll_rate"] == 0.1
    assert pt["timeout"] == pt["timeout_rate"] == 0.2
    assert pt["false_park"] == 0.05
    assert pt["pol"] == pt["pol_loss"] == 1.1
    assert pt["val"] == pt["val_loss"] == 2.2
    assert pt["aux"] == pt["aux_loss"] == 0.3
    assert pt["dyn"] == pt["dyn_loss"] == 0.4
    assert pt["return"] == pt["mean_return"] == 3.3
    assert pt["dr_scale"] == 0.7
    assert pt["anchor_frac"] == 0.15
    assert pt["std"] == 0.55
    assert pt["sps"] == 99
    assert pt["wall_s"] == 12.3


def test_parse_metrics_jsonl_long_keys_populate_both(tmp_path):
    p = _write_jsonl(
        tmp_path / "run.metrics.jsonl",
        [
            {
                "iter": 7,
                "succ_rate": 0.9,
                "coll_rate": 0.02,
                "timeout_rate": 0.08,
                "pol_loss": 0.11,
                "val_loss": 0.22,
                "aux_loss": 0.33,
                "dyn_loss": 0.44,
                "mean_return": 4.4,
                "dr_scale": 0.3,
            }
        ],
    )
    (pt,) = parse_metrics_jsonl(p)
    assert pt["succ"] == pt["succ_rate"] == 0.9
    assert pt["coll"] == pt["coll_rate"] == 0.02
    assert pt["timeout"] == pt["timeout_rate"] == 0.08
    assert pt["pol"] == pt["pol_loss"] == 0.11
    assert pt["val"] == pt["val_loss"] == 0.22
    assert pt["aux"] == pt["aux_loss"] == 0.33
    assert pt["dyn"] == pt["dyn_loss"] == 0.44
    assert pt["return"] == pt["mean_return"] == 4.4


def test_parse_log_extracts_both_key_sets(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(
        "iter 10 | pol 0.11 | val 0.22 | aux 0.33 | dyn 0.44 | std 0.55 | succ 123 (dr=0.24)\n"
    )
    (pt,) = parse_log(p)
    assert pt["iter"] == 10
    assert pt["succ"] == pt["succ_rate"] == 123.0
    assert pt["pol"] == pt["pol_loss"] == 0.11
    assert pt["val"] == pt["val_loss"] == 0.22
    assert pt["aux"] == pt["aux_loss"] == 0.33
    assert pt["dyn"] == pt["dyn_loss"] == 0.44
    assert pt["std"] == 0.55
    assert pt["dr_scale"] == 0.24
    # no bogus None placeholders for extracted metrics
    for k in (
        "succ",
        "succ_rate",
        "pol",
        "pol_loss",
        "val",
        "val_loss",
        "aux",
        "aux_loss",
        "dyn",
        "dyn_loss",
        "std",
        "dr_scale",
    ):
        assert pt.get(k) is not None


def test_parse_log_colon_separated_dr(tmp_path):
    p = tmp_path / "run.log"
    p.write_text("iter 3 | pol 1.0 | succ 80 (dr: 0.5)\n")
    (pt,) = parse_log(p)
    assert pt["dr_scale"] == 0.5


def test_eval_bundle_ugv_replay_v1_normalizes_frames(tmp_path):
    frame = {
        "t": 1.5,
        "true_pose": [0.1, 0.2, 0.3],
        "delayed_tag": {"x": 0.1, "y": 0.2, "yaw": 0.3},
        "action": [0.1, 0.2, 0.3],
        "shield_clear": True,
        "valid": True,
        "blackout": False,
        "outlier": False,
        "imu_w": 0.01,
        "imu_yaw": 0.3,
        "steps_since_valid": 2,
    }
    bundle = {
        "format": "ugv_replay_v1",
        "run_id": "synthetic",
        "episodes": [{"episode_id": 0, "frames": [frame]}],
    }
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle))
    d = eval_bundle(p)
    assert d["geometry"]["car_w"] == 0.16
    ep = d["episodes"][0]
    f = ep["frames"][0]
    # normalized keys
    assert f["t"] == 1.5
    assert f["pose"] == [0.1, 0.2, 0.3]
    assert f["tag"] == [0.1, 0.2, 0.3]
    assert f["aux_pose"] == [0.1, 0.2, 0.3]
    assert f["valid"] is True
    assert f["blackout"] is False
    assert f["outlier"] is False
    assert f["tag_age"] == 2
    assert f["imu_w"] == 0.01
    assert f["imu_yaw"] == 0.3
    assert f["act"] == [0.1, 0.2, 0.3]
    assert f["shield_blocked"] is False
    # original keys preserved
    assert f["true_pose"] == [0.1, 0.2, 0.3]
    assert f["delayed_tag"] == {"x": 0.1, "y": 0.2, "yaw": 0.3}
    assert f["action"] == [0.1, 0.2, 0.3]
    assert f["shield_clear"] is True
    # computed per-episode fields
    assert ep["duration"] == 1.5
    expected_err = round(math.hypot(0.1, 0.2) * 1000, 1)
    assert ep["final_error"] == expected_err


def test_eval_bundle_preserves_existing_duration_and_final_error(tmp_path):
    bundle = {
        "format": "ugv_replay_v1",
        "episodes": [
            {
                "episode_id": 0,
                "duration": 9.9,
                "final_error": 12.3,
                "frames": [{"t": 1.0, "true_pose": [0.0, 0.0, 0.0]}],
            }
        ],
    }
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle))
    d = eval_bundle(p)
    ep = d["episodes"][0]
    assert ep["duration"] == 9.9
    assert ep["final_error"] == 12.3


def _synth_l7_run(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    d.joinpath("ticks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_l7_bundle_bool_tag_format(tmp_path, monkeypatch):
    rows = [
        {"t": 100.0, "tag": False, "pose": [1.0, 2.0, 0.5], "imu": [0.5, 0.01]},
        {"t": 100.1, "tag": True, "pose": [1.1, 2.1, 0.6], "imu": [0.6, 0.02]},
    ]
    _synth_l7_run(tmp_path, "synth-bool", rows)
    monkeypatch.setattr(replay_mod, "CALIB_RUNS", tmp_path)
    d = l7_bundle("synth-bool")
    frames = d["episodes"][0]["frames"]
    assert frames[0]["valid"] is False
    assert frames[0]["tag"] is None
    assert frames[1]["valid"] is True
    assert frames[1]["tag"] == [1.1, 2.1, 0.6]
    assert frames[0]["pose"] == [1.0, 2.0, 0.5]
    assert frames[1]["imu_w"] == 0.02
    assert frames[1]["imu_yaw"] == 0.6
    assert frames[1]["act"] == [0, 0, 0]


def test_l7_bundle_src_tag_format_with_age_and_gyro(tmp_path, monkeypatch):
    rows = [
        {
            "t": 50.0,
            "src": "tag",
            "pose": [0.5, 0.6, -0.4],
            "est": [0.5, 0.6, -0.4],
            "imu": [-0.4, 0.001],
            "frame_age": 0.08,
            "imu_w_fed": 0.25,
            "act": [-0.4, -0.2, 0.3],
            "reason": "policy",
        },
        {
            "t": 50.1,
            "src": "none",
            "pose": None,
            "est": [0.51, 0.61, -0.4],
            "imu": [-0.4, 0.002],
            "age_s": 0.15,
            "wz": 0.5,
            "act": [0.0, 0.0, 0.0],
            "reason": "policy",
        },
    ]
    _synth_l7_run(tmp_path, "synth-src", rows)
    monkeypatch.setattr(replay_mod, "CALIB_RUNS", tmp_path)
    d = l7_bundle("synth-src")
    frames = d["episodes"][0]["frames"]
    assert frames[0]["valid"] is True
    assert frames[0]["tag"] == [0.5, 0.6, -0.4]
    assert frames[0]["tag_age"] == 0.08
    assert frames[0]["imu_w"] == 0.25
    assert frames[0]["act"] == [-0.4, -0.2, 0.3]
    assert frames[1]["valid"] is False
    assert frames[1]["tag"] is None
    assert frames[1]["tag_age"] == 0.15
    assert frames[1]["imu_w"] == 0.5


def test_l7_bundle_tag_xyz_keys(tmp_path, monkeypatch):
    rows = [
        {
            "t": 10.0,
            "src": "tag",
            "tag_x": 0.1,
            "tag_y": 0.2,
            "tag_yaw": 0.3,
            "est": [0.1, 0.2, 0.3],
            "imu": [0.3, 0.01],
            "imu_wz": 0.05,
        },
    ]
    _synth_l7_run(tmp_path, "synth-xyz", rows)
    monkeypatch.setattr(replay_mod, "CALIB_RUNS", tmp_path)
    d = l7_bundle("synth-xyz")
    (f,) = d["episodes"][0]["frames"]
    assert f["valid"] is True
    assert f["tag"] == [0.1, 0.2, 0.3]
    assert f["imu_w"] == 0.05


@pytest.mark.skipif(
    not (CALIB_RUNS / "20261113-184652-policy-gyro-zero" / "ticks.jsonl").exists(),
    reason="specific local L7 policy run telemetry not present",
)
def test_l7_bundle_real_policy_run_extracts_valid_tags():
    d = l7_bundle("20261113-184652-policy-gyro-zero")
    frames = d["episodes"][0]["frames"]
    assert frames, "expected frames from real L7 policy run"
    valid = [f for f in frames if f["valid"]]
    assert valid, "expected at least one tag-valid frame"
    assert all(f["tag"] is not None for f in valid)
    assert all(isinstance(f["act"], list) and len(f["act"]) == 3 for f in frames)
    assert any(f["tag_age"] is not None for f in frames)
    assert any(f["imu_w"] is not None for f in frames)


def test_leaderboard_gazebo_summary_keys(tmp_path, monkeypatch):
    gz = tmp_path / "gazebo_l6"
    gz.mkdir()
    (gz / "synth_gazebo.json").write_text(
        json.dumps(
            {
                "summary": {
                    "episodes": 100,
                    "success": 97,
                    "collisions": 0,
                    "timeouts": 3,
                    "level": "L1",
                    "time_median": 8.4,
                    "pos_err_p95_m": 0.012,
                    "yaw_err_p95_deg": 1.5,
                },
                "episodes": [{"result": "success", "t": 8.0}],
            }
        )
    )
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    entries = {e["run_id"]: e for e in server_mod._leaderboard()}
    e = entries["synth_gazebo"]
    assert e["episodes"] == 100
    assert e["success"] == 97
    assert e["timeouts"] == 3
    assert abs(e["success_rate"] - 0.97) < 1e-9
    assert e["pos_p95"] == 0.012
    assert e["yaw_p95_deg"] == 1.5
    assert e["wilson_lo"] is not None and e["wilson_hi"] is not None


def test_leaderboard_archive_empty_eval_metrics_not_fake(tmp_path, monkeypatch):
    (tmp_path / "archive.jsonl").write_text(
        json.dumps(
            {
                "run_id": "synth_unevaluated",
                "track": "student",
                "level": "L1",
                "eval_metrics": {},
                "config_delta": {"dr_scale": 0.5},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    entries = {e["run_id"]: e for e in server_mod._leaderboard()}
    e = entries["synth_unevaluated"]
    assert e["success"] is None
    assert e["episodes"] is None
    assert e["success_rate"] is None
    assert e["wilson_lo"] is None
    assert e["wilson_hi"] is None


def test_leaderboard_l7_rows_have_real_wilson_ci():
    entries = server_mod._leaderboard()
    l7 = [e for e in entries if e.get("source") == "l7"]
    assert l7, "expected L7 rows in leaderboard"
    for e in l7:
        assert e["wilson_lo"] is not None and e["wilson_hi"] is not None
        assert e["wilson_hi"] >= e["wilson_lo"]
    # a successful L7 run must show a non-degenerate interval
    successful = [e for e in l7 if e["success"] == 1]
    if successful:
        assert successful[0]["wilson_hi"] > 0.5


def test_contract_unknown_run_id_returns_404():
    r = client.get("/api/contract", params={"run_id": "definitely-not-a-real-run-xyz"})
    assert r.status_code == 404


@requires_l7_data
def test_contract_default_run_id_resolves():
    r = client.get("/api/contract")
    assert r.status_code == 200
    d = r.json()
    assert d.get("sim_source") in ("sim_reference_L1.npz", "nominal_contract")


def test_contract_default_prefers_control_run_over_newer_watch(tmp_path):
    import os

    from tools.observatory import contract

    ticks = "".join(
        json.dumps({"t": 1.0 + 0.02 * i, "tag": True, "pose": [0, 0.5, 0], "imu": [0, 0]}) + "\n"
        for i in range(3)
    )
    for name, mode, mtime in (
        ("20260101-000000-stage", "stage", 1000),
        ("20260102-000000-watch", "watch", 2000),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"args": {"mode": mode}}))
        (d / "ticks.jsonl").write_text(ticks)
        os.utime(d / "ticks.jsonl", (mtime, mtime))
    assert contract.latest_run(tmp_path) == "20260101-000000-stage"
    (tmp_path / "20260101-000000-stage" / "meta.json").write_text(
        json.dumps({"args": {"mode": "watch"}})
    )
    assert contract.latest_run(tmp_path) == "20260102-000000-watch"


def test_eval_bundle_merges_geometry_and_preserves_float_tag_age(tmp_path):
    bundle = {
        "format": "ugv_replay_v1",
        "source": "gazebo_l6",
        "geometry": {"car_w": 0.172, "skid_x": 0.17, "gear_len": 0.47, "gear_r": 0.015},
        "episodes": [
            {
                "episode_id": 0,
                "frames": [
                    {
                        "t": 0.02,
                        "true_pose": [0.0, 1.0, 0.0],
                        "tag_age": 0.08,
                        "steps_since_valid": 4,
                    }
                ],
            }
        ],
    }
    p = tmp_path / "gz.traces.json"
    p.write_text(json.dumps(bundle))
    d = eval_bundle(p)
    assert d["geometry"]["skid_x"] == 0.17
    assert d["geometry"]["gear_len"] == 0.47
    assert d["geometry"]["goal_x"] == 0.03  # default merged in
    assert d["episodes"][0]["frames"][0]["tag_age"] == 0.08


def test_l7_bundle_exposes_camera_frames_and_frame_route(tmp_path, monkeypatch):
    run_dir = _synth_l7_run(
        tmp_path,
        "synth-cam",
        [
            {
                "t": 100.0,
                "src": "tag",
                "pose": [0.0, 0.8, 0.0],
                "est": [0.0, 0.8, 0.0],
                "imu": [0.0, 0.0],
            },
            {"t": 100.2, "src": "none", "pose": None, "est": [0.0, 0.78, 0.0], "imu": [0.0, 0.0]},
        ],
    )
    (run_dir / "frames").mkdir()
    (run_dir / "frames" / "000001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (run_dir / "frames.jsonl").write_text(
        json.dumps(
            {
                "seq": 1,
                "cap": 100.05,
                "jpg": "000001.jpg",
                "tags": [{"id": 14, "how": "det", "c": [100, 100, 140, 100, 140, 140, 100, 140]}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(replay_mod, "CALIB_RUNS", tmp_path)
    monkeypatch.setattr(server_mod, "CALIB_RUNS", tmp_path)
    d = l7_bundle("synth-cam")
    assert len(d["camera_frames"]) == 1
    assert d["camera_frames"][0]["frame_url"] == "/api/l7/synth-cam/frame/000001.jpg"
    assert d["episodes"][0]["frames"][1]["frame_url"] == "/api/l7/synth-cam/frame/000001.jpg"
    r = client.get("/api/l7/synth-cam/frame/000001.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8\xff\xd9"
    bad = client.get("/api/l7/synth-cam/frame/..%2Fticks.jsonl")
    assert bad.status_code in (400, 404)


def test_gazebo_l6_traces_discovery_and_bundle(tmp_path, monkeypatch):
    gz = tmp_path / "gazebo_l6"
    gz.mkdir()
    (gz / "demo_run.traces.json").write_text(
        json.dumps(
            {
                "format": "ugv_replay_v1",
                "source": "gazebo_l6",
                "geometry": {"car_w": 0.172, "gear_len": 0.47},
                "episodes": [
                    {
                        "episode_id": 0,
                        "outcome": "success",
                        "frames": [{"t": 0.0, "true_pose": [0, 1, 0]}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    monkeypatch.setattr(server_mod, "CALIB_RUNS", tmp_path / "l7_none")
    bundles = client.get("/api/replay/list").json()["bundles"]
    item = next(b for b in bundles if b["bundle_id"] == "gazebo_l6:demo_run")
    assert item["source"] == "gazebo_l6"
    assert item["episodes_count"] == 1 and item["success_count"] == 1
    assert item["label"] == "demo_run"
    r = client.get("/api/replay/bundle/gazebo_l6:demo_run")
    assert r.status_code == 200
    assert r.json()["geometry"]["gear_len"] == 0.47


# ---- Observatory v2: backend contract for the new dashboard ----

import tools.observatory.bundles as bundles_mod  # noqa: E402
import tools.observatory.contract as contract_mod  # noqa: E402
import tools.observatory.data as data_mod  # noqa: E402
import tools.observatory.leaderboard as leaderboard_mod  # noqa: E402
import tools.observatory.rollout as rollout_mod  # noqa: E402


def test_bundle_source_classification():
    from tools.observatory.bundles import bundle_source

    assert bundle_source("gazebo_L6_student_L2_cons_adr_s0_L1") == "gazebo_l6"
    assert bundle_source("gazebo_l6:demo") == "gazebo_l6"
    assert bundle_source("l7:20261113-184652-policy-gyro-zero") == "l7"
    assert bundle_source("obs_1794669415") == "sim"
    assert bundle_source("student_L2_cons_adr_s0_L1") == "sim"


def test_bundle_counts_for_trace_and_l7(tmp_path, monkeypatch):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "gazebo_L6_demo.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"outcome": "success", "frames": []},
                    {"outcome": "collision", "frames": []},
                    {"outcome": "success", "frames": []},
                ]
            }
        )
    )
    (traces / "obs_1.json").write_text("{not json")
    run = tmp_path / "l7" / "run_a"
    run.mkdir(parents=True)
    (run / "ticks.jsonl").write_text('{"t": 1.0}\n')
    (run / "summary.json").write_text(json.dumps({"outcome": "parked"}))
    monkeypatch.setattr(bundles_mod, "_STATS_CACHE", {})
    items = {b["bundle_id"]: b for b in bundles_mod.list_bundles(tmp_path, tmp_path / "l7")}
    g = items["gazebo_L6_demo"]
    assert (g["source"], g["episodes_count"], g["success_count"]) == ("gazebo_l6", 3, 2)
    bad = items["obs_1"]
    assert bad["episodes_count"] is None and bad["source"] == "sim"
    l7 = items["l7:run_a"]
    assert (l7["source"], l7["episodes_count"], l7["success_count"]) == ("l7", 1, 1)


def test_resolve_rejects_traversal_and_absolute_paths(tmp_path):
    for bad in ("/etc/passwd", "..%2Fsecret", "a/../../b", "l7:../x", "gazebo_l6:a/b"):
        try:
            bundles_mod.resolve(bad, tmp_path, tmp_path)
        except ValueError:
            continue
        except LookupError:
            continue
        raise AssertionError(f"{bad!r} was resolved")


def test_replay_bundle_absolute_id_is_400_not_500():
    r = client.get("/api/replay/bundle//etc/passwd")
    assert r.status_code in (400, 404)


def test_curve_runs_prefers_jsonl_and_dedupes(tmp_path):
    (tmp_path / "run_a.metrics.jsonl").write_text('{"iter": 0, "pol": 1.0}\n{"iter": 1}\n')
    (tmp_path / "run_a.log").write_text("iter 0 pol 1.0\n")
    (tmp_path / "run_b.log").write_text("iter 0 pol 2.0 succ 3 coll 1\n")
    runs = {r["run_id"]: r for r in data_mod.curve_runs(tmp_path)}
    assert sorted(runs) == ["run_a", "run_b"]
    assert runs["run_a"]["format"] == "jsonl" and runs["run_a"]["has_log"] is True
    assert runs["run_a"]["n_iters"] == 2
    assert runs["run_b"]["format"] == "log"


def test_parse_metrics_jsonl_iter_falls_back_to_line_index(tmp_path):
    p = _write_jsonl(tmp_path / "x.metrics.jsonl", [{"pol": 1.0}, {"pol": 0.5}])
    pts = parse_metrics_jsonl(p)
    assert [pt["iter"] for pt in pts] == [0, 1]


def test_l7_runs_tolerates_corrupt_summary(tmp_path, monkeypatch):
    run = tmp_path / "broken"
    run.mkdir()
    (run / "summary.json").write_text("{truncated")
    (run / "ticks.jsonl").write_text('{"t": 1}\n')
    monkeypatch.setattr(data_mod, "CALIB_RUNS", tmp_path)
    rows = data_mod.l7_runs()
    assert rows and rows[0]["run_id"] == "broken" and rows[0]["outcome"] is None


def test_contract_missing_pass_flag_is_unknown_not_pass():
    checks = contract_mod.norm_checks({"valid_pose": {"value": 0.4, "threshold": 0.5}}, {})
    assert checks[0]["pass"] is None
    checks = contract_mod.norm_checks([{"name": "hz", "status": "FAIL"}], {})
    assert checks[0]["pass"] is False


def test_overview_exposes_kpis():
    d = client.get("/api/overview").json()
    assert {"eval_runs", "gazebo_l6_runs", "l7_runs", "pi_step_ms_p50"} <= set(d["kpi"])


def test_rollout_options_lists_engines():
    d = client.get("/api/rollout/options").json()
    assert set(d["engines"]) == {"sim", "gazebo"}
    assert "baseline" in d["engines"]["gazebo"]["policies"]
    assert "oracle_pd" in d["engines"]["sim"]["policies"]


def test_rollout_sim_command_uses_runs_relative_policy(tmp_path):
    (tmp_path / "student_x.pt").write_bytes(b"x")
    argv, cwd, js, trace = rollout_mod._command("sim", "student_x.pt", "L2", 5, 11, 1234, tmp_path)
    assert cwd == rollout_mod.REPO / "rl"
    assert argv[argv.index("--policy") + 1] == "runs/student_x.pt"
    assert argv[argv.index("--save-traces") + 1] == str(tmp_path / "traces" / "obs_1234.json")
    assert argv[0].endswith("python")


def test_rollout_gazebo_command_baseline_and_weights(tmp_path):
    argv, cwd, _js, trace = rollout_mod._command("gazebo", "baseline", "L1", 5, 11, 99, tmp_path)
    assert cwd == rollout_mod.REPO
    assert "simulation/gazebo/run_policy_episodes.py" in argv
    assert "--bridge" in argv and argv[argv.index("--bridge") + 1] == "gazebo"
    assert "--baseline" in argv and "--weights" not in argv
    assert trace.name == "gazebo_99.json" and trace.parent.name == "traces"
    (tmp_path / "gazebo_l6").mkdir()
    (tmp_path / "gazebo_l6" / "w.npz").write_bytes(b"x")
    argv, *_ = rollout_mod._command("gazebo", "gazebo_l6/w.npz", "L1", 5, 11, 99, tmp_path)
    assert "--weights" in argv and "--baseline" not in argv


def test_rollout_rejects_bad_requests():
    for body in (
        {"engine": "warp", "policy": "brake", "level": "L0"},
        {"engine": "sim", "policy": "../../etc/passwd", "level": "L0"},
        {"engine": "sim", "policy": "brake", "level": "L9"},
        {"engine": "sim", "policy": "brake", "level": "L0", "episodes": "many"},
    ):
        assert client.post("/api/rollout", json=body).status_code == 400, body


def test_rollout_busy_returns_409():
    rollout_mod.JOBS["job_busy_test"] = {"job_id": "job_busy_test", "status": "running"}
    try:
        r = client.post("/api/rollout", json={"policy": "brake", "level": "L0", "episodes": 1})
        assert r.status_code == 409
    finally:
        rollout_mod.JOBS.pop("job_busy_test", None)


def test_rollout_status_unknown_job_is_404():
    assert client.get("/api/rollout/job_does_not_exist").status_code == 404


# ---- Regression tests: probe runs and JSON-named directories must not be ranked ----


def _gz_summary(path, episodes, success, level="L1", collisions=0):
    path.write_text(
        json.dumps(
            {
                "episodes": episodes,
                "success": success,
                "collisions": collisions,
                "level": level,
            }
        )
    )
    return path


def test_counts_skip_json_directories_and_probe_runs(tmp_path, monkeypatch):
    gz = tmp_path / "gazebo_l6"
    gz.mkdir()
    run_dir = gz / "baseline_L1_20ep_s11.json"
    run_dir.mkdir()  # a run directory, not a summary file
    (run_dir / "result.json").write_text(json.dumps({"episodes": 20, "success": 0}))
    _gz_summary(gz / "gate_L1_100ep.json", 100, 97)
    _gz_summary(gz / "dbg_cam_L1.json", 1, 0)
    _gz_summary(gz / "smoke_emu_L1.json", 1, 0)
    _gz_summary(gz / "obs_1794_L1.json", 1, 0)
    (gz / "parallel_backlog_plan.json").write_text('[{"name": "student_L2_a", "episodes": 10}]')
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    assert server_mod._counts()["gazebo_l6"] == 1


def test_leaderboard_excludes_probe_and_tiny_runs(tmp_path, monkeypatch):
    gz = tmp_path / "gazebo_l6"
    gz.mkdir()
    _gz_summary(gz / "gate_L1_100ep.json", 100, 97)
    _gz_summary(gz / "dbg_perfect_L1.json", 20, 20)
    _gz_summary(gz / "smoke_perfect_L1.json", 20, 20)
    _gz_summary(gz / "obs_perfect_L1.json", 20, 20)
    _gz_summary(gz / "live_perfect_L1.json", 20, 20)
    _gz_summary(gz / "tiny_perfect_L1.json", 3, 3)
    (gz / "dir_L1.json").mkdir()
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    entries = {e["run_id"]: e for e in server_mod._leaderboard()}
    assert "gate_L1_100ep" in entries
    for rid in (
        "dbg_perfect_L1",
        "smoke_perfect_L1",
        "obs_perfect_L1",
        "live_perfect_L1",
        "tiny_perfect_L1",
        "dir_L1",
    ):
        assert rid not in entries
    # the probe run's perfect 20/20 must not dominate or win the L1 stage card
    assert entries["gate_L1_100ep"]["pareto"] is True


def test_zero_success_never_earns_pareto(tmp_path, monkeypatch):
    gz = tmp_path / "gazebo_l6"
    gz.mkdir()
    _gz_summary(gz / "fail_only_L1_20ep.json", 20, 0)
    _gz_summary(gz / "gate_L1_100ep.json", 100, 97)
    monkeypatch.setattr(server_mod, "RL_RUNS", tmp_path)
    entries = {e["run_id"]: e for e in server_mod._leaderboard()}
    zero = entries["fail_only_L1_20ep"]
    assert zero["success_rate"] == 0.0
    assert zero["pareto"] is False
    assert "Pareto" not in zero["badges"]
    assert entries["gate_L1_100ep"]["pareto"] is True


def test_mark_pareto_rejects_lone_zero_success_run():
    rows = [
        {
            "run_id": "zero",
            "level": "L1",
            "success": 0,
            "episodes": 20,
            "collisions": 0,
            "success_rate": 0.0,
            "source": "gazebo_l6",
        }
    ]
    leaderboard_mod._mark_pareto(rows)
    assert rows[0]["pareto"] is False
    assert rows[0]["pareto_score"] is None
    assert rows[0]["badges"] == ["gazebo_l6"]
