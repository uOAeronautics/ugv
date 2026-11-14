from __future__ import annotations

from collections import deque
import os
import resource
import threading
import time
from typing import Any


def _percentile(sorted_vals: list[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = pct * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class PerfMonitor:
    """Lightweight, non-blocking rolling stage latency and process stats profiler."""

    BUDGET_MS = 33.33

    def __init__(self, maxlen: int = 60, system_ttl_s: float = 0.5) -> None:
        self._maxlen = maxlen
        self._system_ttl_s = system_ttl_s
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {}
        self._times: dict[str, deque[float]] = {}
        self._sys_at: float = 0.0
        self._sys_cache: dict[str, Any] | None = None
        self._last_cpu_at: float | None = None
        self._last_cpu_time: float = 0.0

    def record(self, stage: str, ms: float) -> None:
        now = time.monotonic()
        val = max(0.0, float(ms))
        with self._lock:
            samples = self._samples.get(stage)
            if samples is None:
                samples = deque(maxlen=self._maxlen)
                self._samples[stage] = samples
                self._times[stage] = deque(maxlen=self._maxlen)
            samples.append(val)
            self._times[stage].append(now)

    def _read_rss_mb(self, ru_maxrss: int) -> float:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as fh:
                parts = fh.read().split()
            if len(parts) >= 2:
                pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return round(pages * page_size / 1048576.0, 2)
        except Exception:
            pass
        return round(float(ru_maxrss) / 1024.0, 2)

    def _read_temp_c(self) -> float | None:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as fh:
                return round(int(fh.read().strip()) / 1000.0, 1)
        except Exception:
            return None

    def system_stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._sys_cache is not None and (now - self._sys_at) < self._system_ttl_s:
                return dict(self._sys_cache)
            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu_time = float(ru.ru_utime + ru.ru_stime)
            cores = max(1, int(os.cpu_count() or 1))
            if self._last_cpu_at is None or now <= self._last_cpu_at:
                cpu_pct = 0.0
            else:
                wall_dt = now - self._last_cpu_at
                cpu_delta = max(0.0, cpu_time - self._last_cpu_time)
                cpu_pct = round(min(100.0, cpu_delta / wall_dt * 100.0 / cores), 2)
            self._last_cpu_time = cpu_time
            self._last_cpu_at = now
            try:
                load1 = round(float(os.getloadavg()[0]), 2)
            except OSError:
                load1 = 0.0
            rss_mb = self._read_rss_mb(ru.ru_maxrss)
            temp_c = self._read_temp_c()
            stats = {
                "cpu_pct": cpu_pct,
                "load1": load1,
                "cores": cores,
                "rss_mb": rss_mb,
                "temp_c": temp_c,
            }
            self._sys_cache = stats
            self._sys_at = now
            return dict(stats)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        system = self.system_stats()
        stages: dict[str, dict[str, float]] = {}
        with self._lock:
            items = [(k, list(self._samples[k]), list(self._times[k])) for k in self._samples]
        for name, vals, times in items:
            if not vals or not times:
                continue
            if (now - times[-1]) > 4.0:
                continue
            svals = sorted(vals)
            hz = 0.0
            if (now - times[-1]) <= 1.5:
                recent = [t for t in times if (now - t) <= 2.0]
                if len(recent) >= 2:
                    span = recent[-1] - recent[0]
                    if span > 0.0:
                        hz = round((len(recent) - 1) / span, 2)
            stages[name] = {
                "last_ms": round(vals[-1], 3),
                "p50_ms": round(_percentile(svals, 0.50), 3),
                "p95_ms": round(_percentile(svals, 0.95), 3),
                "hz": hz,
            }
        return {"system": system, "stages": stages, "budget_ms": self.BUDGET_MS}
