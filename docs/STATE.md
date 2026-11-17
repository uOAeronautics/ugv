# STATE

Current operational state. For SSH/IP details see [`docs/NETWORK.md`](NETWORK.md). Consolidated
2026-11-17: the live content of `HANDOFF.md` (deleted, recoverable via git) now lives here.

- **Historical snapshots (`2026-08-29`..`2026-10-17`)**: [`docs/archive/STATE_HISTORY.md`](archive/STATE_HISTORY.md).
- **Superseded Dream-RSI failure-replay proposal**: [`docs/archive/FAILURE_REPLAY.md`](archive/FAILURE_REPLAY.md).
- **Decision history, dead ends, open questions and 2026-11-17 corrections**: [`explore/HISTORY.md`](../explore/HISTORY.md).
- **Roadmap run and hypothesis status**: [`explore/ROADMAP.md`](../explore/ROADMAP.md) and `explore/HYPOTHESES.jsonl`.

## Pi pause & deploy state (paused — charging)

- **Pause (2026-11-13)**: `adv034` (`advanced@192.168.8.100`, ssh `pi`) paused for battery
  charging — **do not SSH, deploy or move the car until charging is complete.** Safety stays
  S0 (`docs/SAFETY.md`); any Auto Park / policy-mode run needs the S4 gate (wheels off the
  ground or a clear zone, a human present, a timeout) and the car free. Confirm a parked
  car with the end frame, not an estimate.
- **Last known pose** (read-only watch run `20261113-193354-watch-diag`): stationary
  (`armed=false`, all wheels 0), policy frame: `x=+0.741 m, y=-0.052 m, yaw=-85.3 deg`,
  facing Tag 6 (`0.62 m`); Tags 14 and 3 not in FOV.
- **Pi is behind ferpc** at `2fefd95`; `ferpc` `main` is at least `1b2590d` plus the
  2026-11-14 `PerfMonitor` (`9edc212`+). Next deploy runbook: sync latest `main` to
  `pi:~/ugv` (**preserve `~/ugv/runs/student.npz` and
  `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf`**), restart
  `ugv-dashboard.service`, verify `GET /api/state`, then the S4 gate before policy-mode
  runs. `1b2590d` changes deployed `StudentPolicy` behaviour during tag loss
  (dwell 1.5 s, ±100 deg yaw sweep, zero translation): treat as a hardware change.
- **Checks before every commit**: `uv run ruff check . && uv run ruff format --check . &&
  uv run pytest -q` (root); `cd rl && uv run pytest -q test_rl.py` when `rl/` changed.
- **2026-11-17**: explore roadmap run; car untouched, nothing deployed, S0 throughout.
  See [`explore/HISTORY.md`](../explore/HISTORY.md) §4 and [`explore/ROADMAP.md`](../explore/ROADMAP.md).

## Current verified hardware & software

- **Target**: Raspberry Pi 4B (`adv034`), Debian 13 (`aarch64`), user `advanced`, repo `/home/advanced/ugv`, service `ugv-dashboard.service` (`.venv/bin/python`, port `8765`). See [`docs/NETWORK.md`](NETWORK.md) for SSH.
- **Camera**: Arducam UC-788 mono global-shutter (`1280x720@30`), nominal `fx=fy=968`, deployed override `UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8`.
- **IMU**: Arduino Nano 33 BLE Sense Rev2 (BMI270) over USB (`/dev/ttyACM0`), `~100 Hz` JSON.
- **Car PCB**: Freenove PCB V3.0 (`i2c-1`: `0x40` PCA9685, `0x48` ADC, `0x70` all-call); SPI/LEDs unused.
- **L7 & Auto Park**: [`docs/L7_FIELD_NOTES.md`](L7_FIELD_NOTES.md) for 2026-11-13 real-car L7 runs; pending Pi deploy items are listed above.
- **Baseline & RL**: StagedBaseline v2 reaches `1000/1000` L0, **~986–989/1000 L1** (2026-11-17 working band, `explore/HISTORY.md` §4), `604/1000` L2 (unverified figure, kept as recorded); best L1 student `966/1000` (`rl/runs/gazebo_l6/student_L1_best.npz`). See [`explore/HISTORY.md`](../explore/HISTORY.md) and [`rl/PLAN.md`](../rl/PLAN.md).
- **Archive note**: `HANDOFF.md` was deleted 2026-11-17 after its live content was
  consolidated here; recover it with `git show HEAD:HANDOFF.md`. Decisions live in `explore/`.
