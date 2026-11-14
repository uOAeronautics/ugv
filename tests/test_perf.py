from __future__ import annotations

import json
import resource

import cv2
import numpy as np

from ugv.autopark import TagDetector
from ugv.perf import PerfMonitor
from ugv.web import RobotController


def test_perf_monitor_record_percentiles_and_maxlen() -> None:
    mon = PerfMonitor(maxlen=60)
    for i in range(1, 21):
        mon.record("decode_ms", float(i))
    snap = mon.snapshot()
    assert snap["budget_ms"] == 33.33
    st = snap["stages"]["decode_ms"]
    assert st["last_ms"] == 20.0
    assert st["p50_ms"] == 10.5
    assert abs(st["p95_ms"] - 19.05) < 1e-3

    for i in range(100):
        mon.record("rolling_ms", float(i))
    rst = mon.snapshot()["stages"]["rolling_ms"]
    assert rst["last_ms"] == 99.0
    assert rst["p50_ms"] == 69.5
    assert rst["p95_ms"] >= 96.0


def test_perf_monitor_hz_and_staleness(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("ugv.perf.time.monotonic", lambda: now[0])
    mon = PerfMonitor()
    mon.record("plan_ms", 1.0)
    assert mon.snapshot()["stages"]["plan_ms"]["hz"] == 0.0

    for _ in range(5):
        now[0] += 0.1
        mon.record("plan_ms", 1.5)
    assert mon.snapshot()["stages"]["plan_ms"]["hz"] == 10.0

    now[0] += 1.4
    assert mon.snapshot()["stages"]["plan_ms"]["hz"] == 10.0

    now[0] += 0.2  # > 1.5s since last sample at 100.5
    assert mon.snapshot()["stages"]["plan_ms"]["hz"] == 0.0


def test_perf_monitor_system_stats_throttle(monkeypatch) -> None:
    now = [10.0]
    calls = [0]
    real_getrusage = resource.getrusage

    def counted_rusage(who):
        calls[0] += 1
        return real_getrusage(who)

    monkeypatch.setattr("ugv.perf.time.monotonic", lambda: now[0])
    monkeypatch.setattr("ugv.perf.resource.getrusage", counted_rusage)
    mon = PerfMonitor(system_ttl_s=0.5)

    s1 = mon.system_stats()
    assert set(s1.keys()) == {"cpu_pct", "load1", "cores", "rss_mb", "temp_c"}
    assert s1["cores"] >= 1
    assert s1["rss_mb"] > 0.0
    assert s1["cpu_pct"] >= 0.0
    assert s1["temp_c"] is None or isinstance(s1["temp_c"], float)
    assert calls[0] == 1

    now[0] += 0.2
    mon.system_stats()
    assert calls[0] == 1

    now[0] += 0.4
    mon.system_stats()
    assert calls[0] == 2


def test_tag_detector_last_timings() -> None:
    det = TagDetector()
    gray = np.full((480, 640), 220, dtype=np.uint8)
    dictionary = det.detector.getDictionary()
    marker = cv2.aruco.generateImageMarker(dictionary, 14, 120)
    gray[100:220, 260:380] = marker

    res_gray = det.detect_gray(gray, 1.0)
    assert res_gray["cap"] == 1.0
    assert set(det.last_timings.keys()) == {"detect_ms", "subpix_ms", "pnp_ms"}
    assert det.last_timings["detect_ms"] >= 0.0
    assert det.last_timings["subpix_ms"] >= 0.0
    assert det.last_timings["pnp_ms"] >= 0.0

    ok, buf = cv2.imencode(".jpg", gray)
    assert ok
    res_jpeg = det.detect_jpeg(buf.tobytes(), 2.0)
    assert res_jpeg["cap"] == 2.0
    assert set(det.last_timings.keys()) == {"decode_ms", "detect_ms", "subpix_ms", "pnp_ms"}
    assert det.last_timings["decode_ms"] >= 0.0


def test_robot_controller_payload_perf_structure() -> None:
    ctrl = RobotController()
    ctrl.perf.record("decode_ms", 1.25)
    payload = ctrl.payload()
    assert "perf" in payload["state"]
    perf = payload["state"]["perf"]
    assert perf["budget_ms"] == 33.33
    assert set(perf["system"].keys()) == {"cpu_pct", "load1", "cores", "rss_mb", "temp_c"}
    assert "decode_ms" in perf["stages"]
    assert set(perf["stages"]["decode_ms"].keys()) == {"last_ms", "p50_ms", "p95_ms", "hz"}
    json.dumps(payload)


def test_perf_monitor_cpu_pct_normalized_by_cores(monkeypatch) -> None:
    now = [100.0]
    utime = [1.0]

    class FakeRu:
        ru_stime = 0.0
        ru_maxrss = 32768

        @property
        def ru_utime(self) -> float:
            return utime[0]

    monkeypatch.setattr("ugv.perf.time.monotonic", lambda: now[0])
    monkeypatch.setattr("ugv.perf.os.cpu_count", lambda: 4)
    monkeypatch.setattr("ugv.perf.resource.getrusage", lambda who: FakeRu())

    mon = PerfMonitor(system_ttl_s=0.1)
    s0 = mon.system_stats()
    assert s0["cpu_pct"] == 0.0
    assert s0["cores"] == 4

    now[0] += 1.0
    utime[0] += 2.0  # 2.0s CPU in 1.0s wall across 4 cores -> 50.0%
    s1 = mon.system_stats()
    assert s1["cpu_pct"] == 50.0

    now[0] += 1.0
    utime[0] += 10.0  # saturated past 400% -> clamped to 100.0%
    s2 = mon.system_stats()
    assert s2["cpu_pct"] <= 100.0
    assert s2["cpu_pct"] == 100.0


def test_perf_monitor_stale_stage_expiry(monkeypatch) -> None:
    now = [200.0]
    monkeypatch.setattr("ugv.perf.time.monotonic", lambda: now[0])
    mon = PerfMonitor()
    mon.record("plan_ms", 2.5)
    mon.record("broadcast_ms", 1.0)
    assert "plan_ms" in mon.snapshot()["stages"]
    assert "broadcast_ms" in mon.snapshot()["stages"]

    now[0] += 3.9
    mon.record("broadcast_ms", 1.2)
    snap_mid = mon.snapshot()
    assert "plan_ms" in snap_mid["stages"]
    assert "broadcast_ms" in snap_mid["stages"]

    now[0] += 0.2  # plan_ms is now 4.1s idle (> 4.0s), broadcast_ms is 0.2s old
    snap_aged = mon.snapshot()
    assert "plan_ms" not in snap_aged["stages"]
    assert "broadcast_ms" in snap_aged["stages"]


def test_aruco_follower_records_aruco_ms(monkeypatch) -> None:
    import asyncio
    import time

    from ugv.aruco import ArucoFollower

    class FakeCamera:
        def latest_sample(self):
            return b"fake-jpeg", 1, time.monotonic()

    ctrl = RobotController()

    async def noop_broadcast():
        return None

    monkeypatch.setattr(ctrl, "broadcast", noop_broadcast)
    follower = ArucoFollower(FakeCamera(), ctrl)
    follower.enabled = True
    monkeypatch.setattr(follower, "track_all", lambda frame: [])

    async def run_once() -> None:
        task = asyncio.create_task(follower.run())
        try:
            for _ in range(100):
                if "aruco_ms" in ctrl.perf.snapshot()["stages"]:
                    break
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_once())
    stages = ctrl.perf.snapshot()["stages"]
    assert "aruco_ms" in stages
    assert stages["aruco_ms"]["last_ms"] >= 0.0
