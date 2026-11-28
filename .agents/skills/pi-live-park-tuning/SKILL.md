---
name: pi-live-park-tuning
description: Deploy, pull telemetry, diagnose, and tune live Auto Park on the Raspberry Pi 4B rover (adv034). Use when iterating on parking control, yaw/strafe stability, vision-pose lock, or field debugging over SSH/WireGuard.
---

# Live Pi Auto Park Tuning (`adv034`)

## 1. Access & fast safe deploy

- Read `docs/STATE.md` and `docs/SAFETY.md` before touching the Pi. Never alter `/etc/systemd/system/ugv-dashboard.service.d/vision-camera.conf` or motor safety invariants (`600 ms` watchdog, `1800/4095` PWM cap, `Space`/`Stop`/`Disarm` brakes).
- Jump path from `ferpc`: `ssh mac "ssh advanced '...'"`. If SSH/SCP stalls on public/makerspace Wi-Fi (PMTU blackhole), clamp WireGuard MTU to `1280` and TCP MSS to `1240`.
- Fast file deploy (after local `ruff check`, `ruff format --check`, and `pytest`):
  ```sh
  tar czf - <changed files> | ssh mac "ssh advanced 'tar xzf - -C ~/ugv && sudo systemctl restart ugv-dashboard.service && sleep 2 && systemctl is-active ugv-dashboard.service'"
  ```

## 2. Log-first diagnosis (never guess)

Every Auto Park run records 50 Hz `telemetry.jsonl`, `summary.json`, and ~3 Hz `1280×720` `frames/*.jpg` to `adv034:~/ugv/runs/autopark/<YYYYMMDD-HHMMSS>/` (auto-pruned to 12 runs). Pull and summarize immediately:

```sh
uv run python tools/park/inspect_pi_run.py [--run <id>] [--frames]
```

Also inspect live runs in the dashboard's **FSM Inspector** (`Audit` button): active gate bars, 12 s oscilloscope (`w_cmd` vs `gyro`, `y cm` vs `±lat_tol`, `duty`), and persistent transition log.

## 3. Sign conventions (memorize before editing)

| Signal | `ugv.park` body frame | `web.py` / hardware |
|---|---|---|
| Forward | `+vx` forward | `forward = duty[0] / DUTY_CAP` |
| Strafe | `+vy` **LEFT** | `strafe = -duty[1] / DUTY_CAP` (**negated**: `+strafe` is RIGHT) |
| Yaw | `+w` CCW (left) | `turn = duty[2] / DUTY_CAP` (`+turn` is CCW) |
| Gyro | `gyro_ccw_rad_s` CCW+ | `RelativeHeading.heading_rate_dps` is **CW+** → negate! |

## 4. Measured real-car control lessons (`adv034` on concrete)

- **Yaw PI & phase lag**: the Pi runs at ~32–40 Hz with ~80 ms motor+IMU phase lag.
  - Active turning (`|w_cmd| >= 0.08`): `kp = 0.05`, `ki = 0.18`, duty cap `±0.28` (~1140 PWM). Higher gains (`kp >= 0.10`, `k_face >= 2.5`) cause 60+ sign-flip jitter and `3 rad/s` spikes.
  - Anti-veer damping while translating (`|w_cmd| < 0.08`): `kp = 0.04`, `ki = 0.06`, `0.15 rad/s` deadband, cap `±0.12`. Setting `kp = 0.14` triggers a textbook 4 Hz limit cycle.
- **Mecanum breakaway friction**:
  - Sideways roller friction is higher than forward (`deadzones = (0.11, 0.17, 0.11)`).
  - Never taper forward duty below ~`0.13` (~530 PWM) when commanding motion, or `APPROACH` stalls right before `stage_x`.
  - Translation stall booster (`_trans_boost`, up to `1.65×` after `0.25 s` of `< 1.5 cm` motion) **must reset to `1.0` on any direction reversal or stage change**, or it blasts across the centerline.
- **Keeping the plane in FOV during `ORBIT`**:
  - Use orbital feedforward $\omega_{\text{ff}} = (x v_y - y v_x) / (x^2 + y^2)$ with calm `k_face = 1.3`, and slow translation to `0.35×` if `|err_face| > 18°`.
  - In `SEARCH`, pause `0.40 s` before spinning so the first `~150 ms` YOLO frame lands while still; if pose `q` is known, aim via shortest IMU angle and scan CW (`-search_w`) from the right side (`y < 0`) or CCW (`+search_w`) from the left (`y >= 0`).
- **Bay entry & pose solver**:
  - Use continuous 50 Hz `ENTER` (`continuous_enter=True`) + direct `gearN` visual-servo centering (`u = 640 px`: pause forward and pure-strafe if `|err_u| > 40 px`, blend if `> 18 px`).
  - Real-world CAD-vs-airframe gear flex is ~10–14 px: keep planar `gate >= 14.0 px` and allow 3–5 pt cold-start floor-plane solves (`hubRear + gearL + gearR`) via 8-heading search in `ugv.park.perception.solve`.
