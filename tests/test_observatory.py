import json
import math
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.observatory.replay as replay_mod
import tools.observatory.server as server_mod
from tools.observatory.data import parse_log, parse_metrics_jsonl
from tools.observatory.replay import eval_bundle, l7_bundle
from tools.observatory.server import app

client = TestClient(app)


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


def test_contract_default_run_id_resolves():
    r = client.get("/api/contract")
    assert r.status_code == 200
    d = r.json()
    assert d.get("sim_source") in ("sim_reference_L1.npz", "nominal_contract")
