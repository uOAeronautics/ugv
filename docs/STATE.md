# STATE

Current state of the car and the Pi: the only file that says whether the car may be touched.
SSH and IP: [`NETWORK.md`](NETWORK.md). Work in progress, decisions and past attempts:
`research/` (`uv run python tools/research/ledger.py status`). Dated snapshots before
2026-10-17: [`archive/STATE_HISTORY.md`](archive/STATE_HISTORY.md).

## Pi pause & deploy state (paused — charging)

- **Pause (2026-11-13)**: `adv034` (`advanced@192.168.8.100`, ssh `pi`) paused for battery
  charging — **do not SSH, deploy or move the car until charging is complete.** Safety stays
  S0 (`docs/SAFETY.md`); any Auto Park or closed-loop run needs the S4 gate (wheels off the
  ground or a clear zone, a human present, a timeout) and the car free. Confirm a parked
  car with the end frame, not an estimate.
- **Last known pose** (read-only watch run `20261113-193354-watch-diag`): stationary
  (`armed=false`, all wheels 0), policy frame: `x=+0.741 m, y=-0.052 m, yaw=-85.3 deg`,
  facing Tag 6 (`0.62 m`); Tags 14 and 3 not in FOV.
- **Pi is behind ferpc** at `2fefd95`; `ferpc` `main` is at least `1b2590d` plus the
  2026-11-14 `PerfMonitor` (`9edc212`+). Next deploy runbook: sync latest `main` to
  `pi:~/ugv` (**preserve
  `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf`**), restart
  `ugv-dashboard.service`, verify `GET /api/state`, then the S4 gate before any closed-loop
  run. Since 2026-11-17 `main` has no RL controller: the dashboard offers `manual` and
  `staged_ekf` only, and the L7 harness `probe`, `stage` and `watch`. That removal changes what
  the car can run, so the next deploy is a hardware change.
- **2026-11-17**: 2D RL stack retired (`research/decisions/2026-11-17-retire-2d-rl.md`) and
  its controller removed from `main`; the Pi still runs the old code until the next deploy.
  Car untouched, nothing deployed, S0 throughout.

## Current verified hardware & software

- **Target**: Raspberry Pi 4B (`adv034`), Debian 13 (`aarch64`), user `advanced`, repo `/home/advanced/ugv`, service `ugv-dashboard.service` (`.venv/bin/python`, port `8765`). See [`docs/NETWORK.md`](NETWORK.md) for SSH.
- **Camera**: Arducam UC-788 mono global-shutter (`1280x720@30`), nominal `fx=fy=968`, deployed override `UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8`.
- **IMU**: Arduino Nano 33 BLE Sense Rev2 (BMI270) over USB (`/dev/ttyACM0`), `~100 Hz` JSON.
- **Car PCB**: Freenove PCB V3.0 (`i2c-1`: `0x40` PCA9685, `0x48` ADC, `0x70` all-call); SPI/LEDs unused.
- **L7 & Auto Park**: [`docs/L7_FIELD_NOTES.md`](L7_FIELD_NOTES.md) for 2026-11-13 real-car L7 runs; pending Pi deploy items are listed above.
- **Past sim and car results** (RL, baseline, Gazebo): `research/attempts/`, roots
  `2026-11-17-track-2d-rl` and `2026-11-17-track-explore-loop`.
