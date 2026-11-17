# H-0004 audit: is the L_real failure a frame or sign error?

Scope: read-only, offline. No sim runs, no GPU (CUDA_VISIBLE_DEVICES unset and no torch run), no car, no Pi, no ssh, no code edits, no commit. `explore.py start/close` not run. Source runs read: `rl/runs/sensitivity_L1ft_Lreal.json` (config run id `student_L1_latw4_ft`, level `L_real`, seed 11, 16 episodes, horizon 1.0 s, delta 0.3).

## Verdict

**All conventions match in sign and axis along the approach. No sign or axis mismatch explains the L_real failure.**

- Export, deployed NumPy forward, and the sim's history frame agree (same 13-feature order, same oldest-to-newest time order).
- The sim's body-velocity sign for `act[1]` (+ = body +y = away from the drone) is enforced in both the real-plant branch (`rl/sim.py:1108, 1122`) and the posterior branch (`rl/sim.py:449`).
- The Pi maps the same action with the same sign (`tools/l7/l7_run.py:677-682`, `src/ugv/web.py:649-656`).
- So `-dy` means "drive toward the drone" (decreasing y, forward along the car's heading) in the sim, in the shield, and on the Pi. The sensitivity `progress` is also defined toward the drone (`rl/action_sensitivity.py:89`).

Three items are sign-adjacent. None changes the L_real result, but two should be fixed or verified:

1. The sim's tag-bearing sign is the negative of the CCW convention (`rl/sim.py:1201`). It only feeds the FOV gate (`rl/sim.py:1210`), so it has no behavioural effect. Cosmetic, but it contradicts the file's own docstring.
2. `policy.step` receives a CCW yaw inside a `parking.Pose2`, which is CW by definition (`src/ugv/policy.py:159, 232`; `tools/l7/l7_run.py:1343`). The arithmetic is internally consistent, but this breaks the boundary rule in CLAUDE.md (`pose2_from_ccw` / `pose2_to_ccw`).
3. The Pi's yaw sign relies on `--gyro-sign 1.0` with a raw integrated `gz` (`tools/l7/l7_run.py:136-138, 845`; every `calibration/runs/l7/*/meta.json` records `gyro_sign: 1.0`). Whether `gz` is CCW-positive on the chassis cannot be settled offline. This affects the Pi, not the sim-only L_real result.

What the audit does **not** show: a learning limit. The audit rules out a sign or axis flip in the action chain. It does not explain why the policy stalls.

## Premise check on the "+0.25 m" number

`sensitivity_L1ft_Lreal.json` does not show +0.25 m for `-dy` as an absolute progress:

| regime | `-dy` progress (m) | `-dy` coll | `+dy` progress (m) | matrix y-row dprogress |
|---|---|---|---|---|
| mouth | +0.0816 | 0.75 | -0.1717 | -0.2533 |
| blind | +0.1302 | 0.0625 | -0.1325 | -0.2627 |

The -0.2533 entry is `mean[progress(+dy) - progress(-dy)]`, i.e. `-dy` is 0.25 m *more* progress than `+dy`, not 0.25 m of progress by itself. The absolute `-dy` progress is 0.08 m (mouth) and 0.13 m (blind), and in the mouth regime that push collides in 75% of snapshots. The sign direction is right; the failure is clearance (driving straight in at x about 0 into the gear), not reversed motion.

The closed-loop `policy` branch is a different story: progress 0.0084 m (mouth), 0.0002 m (blind), with `mean_abs_action` 0.55 / 0.53, so the policy outputs large actions and does not advance. The same student evaluated at L_real gets 0/1000 success (`rl/runs/gate_sweep_L_real_s11.json`, timeout 0.997), against 966/1000 at L1 (`rl/runs/student_L1_latw4_ft.json`). Those runs are facts for the steward; this audit does not explain them.

## Convention chain (file:line)

### 1. Export and deployed forward

| item | location | convention |
|---|---|---|
| export parity | `rl/export.py:24-30` | Random `N(0,1)` histories (batch 4, seed 0), torch `Student` (float32) vs `StudentPolicy.forward` (float64), max abs error. |
| parity gate | `rl/export.py:52-54` | `assert err < 1e-5`. Export aborts on a drift. |
| exported keys | `rl/export.py:47-51` | Only `gru.*` and `head.*`; aux/dyn/fwd heads excluded. |
| GRU layout | `rl/models.py:70` | `nn.GRU(obs_dim, h, batch_first=True)`: input `[B, HIST, DIM]`, time index 0 = oldest. |
| deployed forward | `src/ugv/policy.py:138-149` | Iterates `t = 0 .. HIST-1` (oldest first), then `_head`. |
| history push | `src/ugv/policy.py:266-268` | `hist[-1]` = newest. |
| stateless step | `src/ugv/policy.py:226` | `act = self.forward(self.hist).clip(-1, 1)` (the path the parity test covers). |
| stateful step | `src/ugv/policy.py:221-223` | `_step_gru` one-step path. **Not covered by export parity.** |
| frame spec | `src/ugv/policy.py:17-18, 189-202` | `DIM=13`: tag(3), age, valid, imu_w, imu_yaw, prev_act(3), tag_vel(3). |
| sim frame | `rl/sim.py:204`, `rl/sim.py:1351-1361` | `STUDENT_DIM=13`; same order: tag, `hist_age` clamp 5, valid, imu_w, `imu_yaw_w` (wrapped), prev_act, tag_ema. |
| sim history order | `rl/sim.py:1363` | `hist = cat([hist[:,1:], frame])`: newest last. Matches `policy.py:266-268`. |
| sim prev_act | `rl/sim.py:1365` | `prev_act = cmd` (clamped action, policy units). Matches `policy.py:198`. |
| sensitivity inference | `rl/action_sensitivity.py:108` | `model(obs["student"])` on the full 50-step history: same path the parity test covers. |

**How parity was checked.** `export.py:24-30` compares the torch module with the deployed NumPy `forward` on the same random histories. Both consume the same array in the same time order, so an ordering error common to both is *not* caught here; the frame ordering was checked by reading (`sim.py:1363` vs `policy.py:266-268`). The recorded value `8.17e-08` appears in `explore/HISTORY.md:35` and the A board copy; I did not re-run `export.py`, so I cannot confirm the number independently. It is consistent with the float32-vs-float64 gap for a 1e-5 gate.

### 2. Frame: policy vs sim (yaw CCW+)

| item | location | convention |
|---|---|---|
| sim frame doc | `rl/sim.py:3-5` | Drone at origin, skids along y, car at `yaw=0` faces `-y` (toward the drone), CCW+ yaw. |
| sim pose integration | `rl/sim.py:1181-1185` | `x += dt(c vx - s vy)`, `y += dt(s vx + c vy)`, `yaw += dt w`. Body `vx`=left (+x at yaw 0), body `vy` = +y at yaw 0 (away from drone). |
| sim IMU yaw | `rl/sim.py:1306` | `imu_yaw += imu_w dt`: CCW, same sign as `vel[2]`. |
| sim tag observation | `rl/sim.py:1196, 1257` | `tag = delayed + noise` where `delayed` is the car pose `(x, y, yaw)` in the drone frame, not a car-relative bearing. |
| sim bearing | `rl/sim.py:1201-1202` | `bearing = atan2(x, y) + yaw`. **Numerically the negative of the CCW-positive bearing** of the drone from the car heading. Checked numerically (below). |
| bearing use | `rl/sim.py:1210` | `in_fov = |bearing| < FOV_HALF & dist < 5`. Sign-insensitive. No other use. |
| deployed shield | `src/ugv/policy.py:232-241` | Uses `estimate.yaw` as CCW, rotates body velocity `(act0 V_X, act1 V_Y)` exactly as `sim.py:1183-1184`. Matches. |
| shield CW conversion | `src/ugv/policy.py:253-257` | Negates yaw for `swept_clear` (CW Pose2), consistent with `parking.pose2_from_ccw` (`src/ugv/parking.py:82-84`). |
| boundary mislabel | `src/ugv/policy.py:159` (annotation `Pose2`), `tools/l7/l7_run.py:1343` (`Pose2(e[0], e[1], e[2])`) | CCW values passed in a CW-typed `Pose2`, bypassing `pose2_from_ccw`. Numerically consistent inside `policy.py`; a boundary violation (CLAUDE.md "Frames and tags"). |
| Pi car frame | `tools/l7/l7_run.py:514-515, 589-591` | `th = yaw - pi/2`, forward `f=(cos th, sin th)`, left `(-sin th, cos th)`: identical to sim forward/left at yaw 0 and CCW. |
| Pi yaw source | `tools/l7/l7_run.py:136-138, 845, 959` | `yaw += sign * radians(gz - bias)`, `sign = --gyro-sign` (default 1.0). Sign of `gz` on the chassis: **not verified offline.** |
| dashboard heading | `src/ugv/heading.py:10` ("clockwise yaw"), `src/ugv/web.py:957-964` | CW heading explicitly negated before `AutoParkRunner.step` (`gyro_yaw_rad = -radians(heading_deg)`, `imu_w_dps = -heading_rate`). |
| GoTo / autopark frame | `src/ugv/autopark.py:451-461`, `368-395` | `th = yaw - pi/2`; `e_f`, `e_l` along forward/left; `yaw_goal = -atan2(x, y)` (heading toward the drone). Same basis as the sim. |

**Numerical check of the bearing sign (read-only inline Python, no file written).** With `forward = (sin yaw, -cos yaw)` and `tgt = atan2(-y, -x)`:

```
x=+0.10 y=0.30 yaw=+0.00  beta_ccw=-0.3218  sim_bearing=+0.3218  sum=+0.0000
x=+0.10 y=0.30 yaw=+0.20  beta_ccw=-0.5218  sim_bearing=+0.5218  sum=+0.0000
x=-0.12 y=0.50 yaw=-0.15  beta_ccw=+0.3855  sim_bearing=-0.3855  sum=+0.0000
x=+0.00 y=0.30 yaw=+0.30  beta_ccw=-0.3000  sim_bearing=+0.3000  sum=+0.0000
```

So `sim_bearing = -beta_ccw` in every case. Only the magnitude gates behaviour (`rl/sim.py:1210`).

### 3. Pi action-to-duty mapping

| item | location | mapping |
|---|---|---|
| action to duty | `tools/l7/l7_run.py:677-682` | `duty_fwd = -act[1]*gain`, `duty_left = act[0]*gain`, `duty_ccw = act[2]*HALF_BASE*4*gain`. Comment: `act[1]` drives +y_drone = backward. |
| duty to dashboard | `tools/l7/l7_run.py:665-674`; `src/ugv/autopark.py:399-405` | `F = fwd/K`, `S = -left/K`, `T = ccw/K`, uniform scaling to the wheel cap. **Duplicated** (identical formula, two files). |
| autopark command | `src/ugv/autopark.py:692-709`, `702` | `forward=F, strafe=S, turn=T`; `last_duty = (F*K, -S*K)` so `propagate` recovers `duty_left = -S*K`. |
| dashboard wheel mixer | `src/ugv/web.py:649-656` | `motor_forward = -forward`; FL `= mf - strafe + turn`, FR `= mf + strafe - turn`, RL `= mf + strafe + turn`, RR `= mf - strafe - turn`. Comment: positive strafe = rightward; positive turn = left (CCW). |
| dashboard write | `src/ugv/web.py:627-635` | Positive wheel value drives `fwd` channel; negative drives `rev`. |
| who runs the student | `tools/l7/l7_run.py:1336-1346` | The only `StudentPolicy.step` caller in `src/` or `tools/` (`web.py` only loads and times the artifact, `web.py:519-525`). The dashboard and autopark do **not** execute the student; they run `GoTo`. |

Resulting sign chain for the student, `act = (a0, a1, a2)`:

- `a1 < 0` gives `duty_fwd > 0` gives `F > 0` gives `motor_forward < 0`: the car drives forward (toward the drone).
- `a0 > 0` gives `duty_left > 0` gives `S < 0`: left.
- `a2 > 0` gives `T > 0`: left turn (CCW), by the `web.py:650` comment and the wheel map `web.py:653-656`.

These are the same signs the sim uses (section 4), so this is a consistency check, not a hardware check. Whether `F > 0` is physically forward depends on the hardware fit (`calibration/`), which I did not re-derive here.

### 4. Sim action to wheel / motion direction vs approach

| item | location | mapping |
|---|---|---|
| level flags | `rl/sim.py:112-188`; `rl/sim.py:287-288` | `real_plant=True` only on `L_chaos` (136), `L_field` (164), `L_real` (187). L0-L5 use the posterior mixer (default False). |
| real plant, u | `rl/sim.py:1108` | `u_raw = [cmd0, -cmd1, cmd2] * dcap` (dcap 0.44). |
| real plant, targets | `rl/sim.py:1121-1124` | `vx = g0 eff0`, `vy = -g1 eff1`, `w = g2 eff2` (+ strafe-yaw coupling). |
| posterior mixer | `rl/sim.py:1134-1137` | `vx, vy, w = cmd0, cmd1, cmd2 HALF_BASE 4`; `wheels = -trans + turn` (wheels from `trans=[vx-vy, vx+vy, vx+vy, vx-vy]`, `turn=[-w,-w,w,w]`). |
| posterior sign filter | `rl/sim.py:433-438, 443, 449-452` | `wheels_for` and the `ok` filter require `dyp[:,1] > 0.15` and `dym[:,1] < -0.15`: `+cmd1` yields `+vy` on every retained particle. |
| pose integration | `rl/sim.py:1181-1185` | As in section 2. |
| strafe-yaw coupling | `rl/sim.py:1124-1131` | Left strafe (`cmd0 > 0`) adds `+rf_strafe_yaw` to `w`. Sign is modelled but not checked against Pi data here. The sensitivity `yaw` metric uses `abs(yaw)` (`action_sensitivity.py:91, 116`), so it cannot show this sign. |

**Resulting sim chain:** `cmd1 < 0` gives `vy < 0` gives `y` decreases gives the car moves toward the drone, in both the real-plant and posterior branches. This matches the Pi (`duty_fwd = -act1*gain`) and the sensitivity `progress = y0 - y_end` (`action_sensitivity.py:89`).

### 5. Sensitivity tool (the source of the "+0.25")

- `rl/action_sensitivity.py:41-46`: `+axis` gives `+delta`, `-axis` gives `-delta`. Branch `-dy` gives `cmd1 = -0.3`.
- `rl/action_sensitivity.py:89`: `progress = y0 - y_end`, positive toward the drone.
- `rl/action_sensitivity.py:75-76`: `coll` is "ever collided within the horizon".
- The 3x3 matrix is `mean[outcome(+axis) - outcome(-axis)]` (`action_sensitivity.py` `matrix` note in the JSON config), so the +0.25 is a contrast, see the premise check above.

## Commands run (all read-only)

```
cat/sed -n ... on rl/export.py, src/ugv/policy.py, rl/sim.py, rl/models.py,
    rl/action_sensitivity.py, src/ugv/autopark.py, src/ugv/web.py,
    src/ugv/heading.py, tools/l7/l7_run.py, src/ugv/parking.py
grep -n / grep -rn (bash -c, for zsh globbing) over rl/, src/, tools/, tests/,
    calibration/, explore/, .steward/ for: yaw/bearing/duty/StudentPolicy/policy.step/gyro_sign/8.17e-08
python3 -c  (JSON summary of rl/runs/sensitivity_L1ft_Lreal.json; keys of rl/runs/student_L1_latw4_ft.json)
python3 -c  (np.load of runs/student.npz: meta_stateful only, read-only)
python3 -   (inline bearing-sign check above; no file written)
```

Not run: `rl/export.py`, `rl/action_sensitivity.py`, any sim, eval, torch load of a checkpoint, GPU, ssh, the car, `explore.py` start/close/predict/propose/retro.

## Not determined / open

- The recorded parity `8.17e-08` was not re-run here (section 1).
- Whether `gz` is CCW-positive on the chassis and whether `F > 0` is physically forward are hardware facts. They live in `calibration/` and the Pi logs, not in this offline audit.
- The stateful `_step_gru` path (`src/ugv/policy.py:221-223`) is not covered by export parity. `runs/student.npz` has `meta_stateful=False`; `student_L1_latw4_ft` was evaluated with full-history forward (`action_sensitivity.py:108`), so the failure is not on that path.
- The L1 (posterior) vs L_real (real-plant) action maps agree in sign for `cmd1` (`rl/sim.py:449`) but differ in gains, deadzones and yaw coupling. A CPU sensitivity run at L1 would confirm the same `-dy` direction under the posterior branch; not run, per the brief.
- Why the L_real closed-loop policy does not advance (progress about 0, timeouts 99.7%) is a learning or regime question, not settled by this audit.
