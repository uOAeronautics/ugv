# STATE

Current operational state. For SSH/IP details see [`docs/NETWORK.md`](NETWORK.md). For live pause/sync details see [`HANDOFF.md`](../HANDOFF.md).

- **Historical snapshots (`2026-08-29`..`2026-10-17`)**: [`docs/archive/STATE_HISTORY.md`](archive/STATE_HISTORY.md).
- **Superseded Dream-RSI failure-replay proposal**: [`docs/archive/FAILURE_REPLAY.md`](archive/FAILURE_REPLAY.md).

## Current verified hardware & software

- **Target**: Raspberry Pi 4B (`adv034`), Debian 13 (`aarch64`), user `advanced`, repo `/home/advanced/ugv`, service `ugv-dashboard.service` (`.venv/bin/python`, port `8765`). See [`docs/NETWORK.md`](NETWORK.md) for SSH.
- **Camera**: Arducam UC-788 mono global-shutter (`1280x720@30`), nominal `fx=fy=968`, deployed override `UGV_CAMERA_SHUTTER_US=4000`, `UGV_CAMERA_GAIN=8`.
- **IMU**: Arduino Nano 33 BLE Sense Rev2 (BMI270) over USB (`/dev/ttyACM0`), `~100 Hz` JSON.
- **Car PCB**: Freenove PCB V3.0 (`i2c-1`: `0x40` PCA9685, `0x48` ADC, `0x70` all-call); SPI/LEDs unused.
- **L7 & Auto Park**: see [`docs/L7_FIELD_NOTES.md`](L7_FIELD_NOTES.md) for 2026-11-13 real-car L7 runs and [`HANDOFF.md`](../HANDOFF.md) for pending Pi deploy items (`1b2590d`+ and live `PerfMonitor`).
- **Baseline & RL**: `StagedBaseline v2` reaches `1000/1000` L0, `989/1000` L1, `604/1000` L2; best L1 student `966/1000` (`rl/runs/gazebo_l6/student_L1_best.npz`). See [`docs/AGENT_BOARD.md`](AGENT_BOARD.md) and [`rl/PLAN.md`](../rl/PLAN.md).
