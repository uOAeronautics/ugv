# UGV Research Observatory — UI/UX Audit & Scientific Dashboard Requirements

**Author:** UI/UX Research Critic 01 — Information Architecture, Typography & Data Visualization
**Date:** 2026-11-14
**Target of fixes:** `tools/observatory/{static/index.html, static/style.css, static/app.js, server.py, replay.py}`
**Builder:** Lead engineer (Haiku 5.5). This document is the spec. Do not gold-plate; fix exactly what is listed.
**Do not touch:** `rl/` training code, `config/`, safety gates, systemd units, or `docs/SAFETY.md` levels. The Observatory is an `S0` read-only surface plus one operator-triggered rollout job; it must never gain motor control.
**Stack decision:** keep vanilla JS + Canvas + FastAPI. No build step, no CDN. The Pi serves this on a LAN; every asset must be local.

> **Verdict.** The current build is a dark-mode-for-the-sake-of-it debug page with the information density of an IDE and the polish of an `<input>` default. It fails on all three axes this dashboard is sold on: legibility (unreadable legend soup, crushed axes), semantics (can't tell a Level pill from a Track pill), and trustworthiness (an `INFO` banner styled as a red failure, a rollout button that dies with `exit 1` and no stderr, empty L6/L7 cards). The fix is a daylight-first engineering theme, an HTML-legend/crosshair chart layer, a rebuilt rollout trigger with real preflight, and first-class Gazebo L6 / STRL001 surface area. Everything below is required unless marked optional.

---

## Part 1 — Defect inventory (grounded in code)

Evidence key: **SS-nn** = `docs/ui_critique/199ad92a-6a9f-4c9f-a319-0ef9d3624c26-<hash>.png` (see Appendix A). File references are to the pre-fix tree.

### 1.1 `static/style.css` — theme does not exist, only a palette dump

| # | Defect | Evidence | Required fix |
|---|---|---|---|
| D-001 | Dark-only `:root` (`#0b0f14`/`#121821`), no light theme, no theme toggle, no `color-scheme` declaration. Native controls render in the OS's light scheme → the “white on dark” clash. | SS-03, SS-06, SS-07, SS-08, SS-11; style.css:1 | Part 2 tokens; `color-scheme: light dark`; default **light**; toggle in topbar. |
| D-002 | Zero scrollbar styling (`.table-wrap`, `.run-selector`, `.controls`, `#episode-list`, native `<select>` popup). Giant OS-default white scrollbars on dark panels — the single loudest “unfinished” signal. | SS-03, SS-07, SS-08, SS-10, SS-11; style.css (no `::-webkit-scrollbar`, no `scrollbar-color`) | Part 2.4 scrollbars. |
| D-003 | `input[type=search],input,select{...}` (line 20) restyles text fields but leaves checkboxes/radios/ranges/native buttons at OS defaults → white checkboxes, white `Run` / `Play` / `-1` / `+1` / `Diff` buttons. | SS-03, SS-06, SS-07; app.js has no `.diff-btn` rule at all (style.css: —) | Part 2.5 form/controls contract. |
| D-004 | No `:focus-visible` styling anywhere → keyboard users get default or nothing. | style.css throughout | Part 8. |
| D-005 | No `font-variant-numeric: tabular-nums` → metric columns jitter on refresh. | style.css:3 | Part 2.2 typography. |
| D-006 | `.banner.fail` used for any non-“pass” status → `INFO` renders as a dark-red error block. | SS-09; app.js:1066 | Tri-state banner, Part 6.3. |
| D-007 | `.badge.pareto` green-on-dark; `.badge.archive` unlabeled gray; `.badge.active` reuses default. No semantic dot/icon system, no text contrast guarantee. | SS-06; style.css:29-31 | Part 2.6 badges. |
| D-008 | `.tab.active` = `#4aa3ff` fill, black text; hover/focus states undefined; tabs carry zero explanatory affordance. | SS-05, SS-06; style.css:10-11 | Part 3 tab IA + Part 2.5 buttons. |
| D-009 | Breakpoint at 1280px collapses charts to 1 column but the 260px sidebar + 440px camera pane still overflow between 1024–1280px. | style.css:90 | Grid: `minmax()` with `overflow:auto`, test at 1024/1280/1440/1920. |
| D-010 | `.chart-card canvas{height:220px}` fixed; chart cards have no unit/label header row, no legend container. | style.css:37-40 | Part 4 layout. |
| D-011 | `--muted:#7a8699` on `#0b0f14` ≈ 4.6:1 (borderline) and unusable on `#1a2230` chips; no contrast contract. | style.css:1 | Part 2.2 contrast ≥ 4.5:1 everywhere. |
| D-012 | `.table-wrap th{position:sticky;top:0}` ok, but no zebra rows, no numeric right-alignment → scanning error rates is needlessly hard. | style.css:27-28 | Part 5.4 table rules. |
| D-013 | No `prefers-reduced-motion`, no `prefers-color-scheme` seeding for first paint. | style.css | Part 8. |

### 1.2 `static/index.html` — structure mirrors the mess

| # | Defect | Evidence | Required fix |
|---|---|---|---|
| D-014 | Header pills are opaque: `Eval runs: 77 (archive 44, gz 13)`, `Pi4B: 0.136 ms/step` — no labels of method, no units beyond terse text, no “as of” timestamp, no tooltips. | SS-05, SS-06, SS-08, SS-09; index.html:14-18 | Part 3.5 header spec. |
| D-015 | Tabs have no subtitles/help; the tab names (`2D Layout & Episode Replay`) don’t say what the tab is *for*. | index.html:22-27 | Part 3.2. |
| D-016 | Filters: `#level-filters` and `#track-filters` are two unlabeled `.filter-group` divs thrown in one `.filters` row → one giant indistinguishable pill soup; `eval_json` and `archive` are **sources** filed as tracks. | SS-06, SS-12; index.html:31-36; app.js:23-24 | Part 5.1 labeled Level / Track / Source clusters. |
| D-017 | `L6` and `L7` cards + `L6` pill: `_leaderboard()` never assigns level `L6` (gazebo runs keep `L1`/`L2`) and assigns L7 runs `level:"L_real"` (server.py:211) → those cards permanently render `—`. Dead UI. | SS-06; server.py:211, app.js:108-113 | Part 5.2 populated L6/L7 cards. |
| D-018 | Curves: 4 fixed chart cards with in-canvas titles only; no axis-unit label, no HTML legend slot, no per-chart empty state text area, canvas is the only interactive surface. | SS-08; index.html:59-65 | Part 4. |
| D-019 | Rollout form: `<option value="student">student</option>` placeholder; policy/level selects populated later; **no engine selector**; “Run” parented to a white native button; raw filenames as options. | SS-03, SS-11; index.html:73-80 | Part 6 (rebuild). |
| D-020 | Bundle `<select>` mixes `student_*` eval bundles, `obs_<ts>` ad-hoc rollouts, `l7:<run>` field runs — flat list, no optgroups, timestamped noise first in list order. | SS-10; app.js:316-323 | Part 7.2 optgroups + search. |
| D-021 | `Rollout Trigger` and `Bundle` are unlabeled section soup; no status region semantics (`role=status`), no disabled state while a job runs. | index.html:70-94 | Part 6.4. |
| D-022 | STRL001 controls (`#target-geometry`, `#view-25d`) exist but are buried, unlabeled as a group, carry no geometry summary, no survey-status badge, and aren’t wired into rollouts. | index.html:94-99; app.js:358-361, 1121-1124 | Part 7 (STRL001 first-class). |
| D-023 | No global help/about affordance; nothing anywhere explains L0→L7, Gazebo, or the contract. | whole file | Part 3.3–3.4. |
| D-024 | `Auto-refresh 5s` checkbox is native white; no last-updated readout; curves tab never refreshes even when enabled. | SS-06; index.html:19; app.js:1134-1136 | Part 3.5 + auto-refresh covers all tabs or is honestly scoped. |
| D-025 | L7 `<details>` section: summary styled by OS default; table inside scrolls; no column for `has_end_jpg`. | index.html:46-53 | Part 5.5. |

### 1.3 `static/app.js` — logic bugs, not just looks

| # | Defect | Evidence | Required fix |
|---|---|---|---|
| D-026 | **Legend overlap root cause (the “garbled string”).** `renderChart` draws legend at fixed `y=4/12`; on overflow it resets `lx=padL+4` **without advancing `y`** (app.js:268-273) → wrapped legend entries draw on top of earlier ones at the same baseline. 6-color palette reused for up to 15 lines (app.js:227, 205). Series names are raw `${rid} · ${label}` — 25–40 chars each × 15 → hundreds of px past canvas width. | SS-08, SS-02; app.js:205, 227, 267-273 | Part 4.2/4.3 — HTML legend outside canvas. |
| D-027 | **Axis crush.** All metrics in a chart share one linear y auto-range (app.js:242-249). `sps ≈ 171,916` vs `std ≈ 0.05–0.3` → std is a flat line at 0. Tick labels use `toFixed(3)` → `171916.559` / `-12734.544` overflow `padL=44` and print over the plot. | SS-08; app.js:238, 242, 255 | Part 4.4 unit-separated charts + dynamic left pad + engineering notation. |
| D-028 | `CHART_SPECS.chart4` plots only `std` and `sps`; title says “std & entropy / mean_return & sps” → entropy and mean_return are **never charted**. `std` is duplicated in chart3. Chart titles lie. | SS-08; app.js:206-211 | Part 4.5 corrected chart registry. |
| D-029 | `chart1` mixes rates (0–1) with `dr_scale` on one axis with no unit label and no legend → which line is “success”? | SS-08 | Split into rate chart + curriculum chart, Part 4.5. |
| D-030 | **Hover tooltip misalignment.** Tooltip indexes other series by *array index* `l.pts[hoverIdx]` (app.js:275-288), but legacy `.log` and `.metrics.jsonl` series have different lengths/iters → values from different iterations are compared. `lines[0].pts[hoverIdx]` assumes series 0 is longest/complete. | SS-08; app.js:275-310 | Part 4.6 iter-synced crosshair via binary search. |
| D-031 | Canvas re-created on every `mousemove`: `c.width=c.clientWidth` clears + full redraw per event (app.js:237, 311-309) → flicker, O(n²) redraws. | app.js:237 | Render-on-demand + rAF throttle, Part 4.7. |
| D-032 | No `devicePixelRatio` scaling → blurry text/lines on any HiDPI display (all modern laptops, Pi HDMI). | app.js:237 | Part 4.7 DPR handling. |
| D-033 | Run-selector checkboxes: `label` innerHTML `<input> <id> (legacy .log)` inline → checkbox wraps to next line / detaches from its row at 220px width; `(legacy .log) [x]` appears to belong to the *next* item. Also unsanitized `innerHTML` with filenames (repo-controlled, still a bug). | SS-02, SS-07, SS-08; app.js:186-190 | Part 7.1 row spec (flex, min-width-0, badge), textContent-only rendering. |
| D-034 | Layers sidebar: same inline-label wrap; checkbox pinned right of nothing coherent; `.layers label{display:block}` (style.css:81) fights the inline input+text flow. | SS-07; app.js:438-441 | Part 7.1. |
| D-035 | `LEVELS` includes `L6` (app.js:23) which no entry ever carries (see D-017) → dead pill. `TRACKS` conflates track (`student/teacher/baseline/gazebo_l6/l7`) with source (`eval_json/archive`). | SS-06, SS-12 | Part 5.1 three labeled groups; drop dead pills. |
| D-036 | Chart colors cycle `ri*spec.length+si` mod 6 (app.js:227) → same color for different metrics across selected runs; no per-run stable hue, no per-metric dash → invisible series identity. | SS-08 | Part 4.3 color/dash mapping. |
| D-037 | `.diff-btn` has **no CSS rule** → white native buttons in the table; `Diff` panel has no units/alignment treatment; `renderDiffPanel` renders raw floats. | SS-06; app.js:125-161 | Part 5.4/5.6. |
| D-038 | `fetchJSON` swallows all errors → failures render as `—` with no message; no toast, no retry. | app.js:36 | Part 3.6 error/empty states. |
| D-039 | `loadCurvesRuns` auto-checks first 3 runs by mtime with a convoluted condition; mixed legacy/modern selection makes D-027/D-030 worse; no cap/warning at >3 runs. | app.js:182-192 | Cap selection, default 1 modern run + hint. |
| D-040 | `state.hover` re-keyed per `drawCharts` but `attachChartHover` binds once at init to canvas elements that persist — fine, but `mousemove` uses `c.width` (device px vs client px mismatch) — recompute against clientWidth & DPR. | app.js:302, 237 | Part 4.7 unified geometry. |
| D-041 | Bundle select `loadBundle` prefers gazebo bundles then falls back to non-`obs_` first entry (app.js:326-331) — accidental ordering, no user-visible grouping; STRL001 geometry only honored when bundle geometry says so, silent otherwise. | app.js:325-331 | Part 7.2. |
| D-042 | Replay canvas text collides at the goal (`ugv`, `tag`, `aux`, `goal`, blind-ring label) — no collision avoidance, fixed offsets (±6, ±12). Cosmetic but it’s in the “hero” viewport. | SS-03; app.js:661, 672, 745, 756, 775 | Offset labels radially with per-frame collision pass or stack them bottom-left as a legend. |
| D-043 | `drawStrips` labels (`imu_w`, `imu_yaw`, `vx vy w`, `valid/...`) drawn in canvas with no units, no legend, no y-scale — the strips are unreadable as data. | SS-03; app.js:944-1013 | Part 4.8 strips spec (HTML captions + units). |
| D-044 | `#btn-play`/`#speed-sel`/`#scrubber`/`-1`/`+1`/`Run` — none styled; playback toggles text Play/Pause on a white button. | SS-03; index.html:87-93 | Part 2.5. |
| D-045 | Auto-refresh only reloads overview+leaderboard; curves/contract/replay go stale silently; no dirty indicator. | app.js:1134-1136 | Refresh all tabs or scope label honestly. |
| D-046 | `renderEpisodeList` default filter `All` with active class wired by ad-hoc classList toggling; `False-Park` filter string matches `false_park` contains “park” — works by accident (`'falsepark'.includes('park')`). | app.js:367-377 | Explicit outcome enum mapping. |
| D-047 | `badge.textContent='GAZEBO CAMERA'` for gazebo bundles but `frame_url`s from a Gazebo trace point at `/api/replay/media/...` which **does not exist** in server.py → camera POV 400/404s with no error text; user sees blank panel. | SS-03; server.py (no `media` route); replay.py:end_frame_url passthrough | Part 7.3 media route or honest empty state. |
| D-048 | `obs_<ts>` rollout bundles appear in Bundle select with no `[obs]` prefix and no link to the job that created them. | SS-10 | Part 7.2 `[rollout]` badge + job id. |

### 1.4 `server.py` — the rollout is broken and the API is silent about it

| # | Defect | Evidence | Required fix |
|---|---|---|---|
| D-049 | **Root cause of `FileNotFoundError: student_L2_cons_adr_s0.pt`.** `/api/rollout` spawns `evaluate.py` with `cwd=rl/` and passes the bare filename from `/api/rollout/options` (`rl/runs/*.pt` → `p.name`, server.py:467). `torch.load("student_L2_cons_adr_s0.pt")` resolves against `cwd`, not `rl/runs/` → always fails for every listed policy. **Every rollout from the UI has failed since this code shipped.** | SS-03 (`job job_1794673578 : failed`); server.py:467, 489-505; evaluate.py:35 | Part 6.2 — server-side path resolution + existence preflight. |
| D-050 | Job failure surfaces as `exit 1` only; stderr is piped nowhere (`asyncio.create_subprocess_exec` without `stderr=PIPE` capture); `JOBS` stores no log tail; UI shows one opaque line. | server.py:489-517, 531-542 | Part 6.3 — capture stderr tail, structured error codes. |
| D-051 | `/api/rollout/options` lists only `*.pt` — although `rl/runs/student_L2_cons_adr_s0.npz` **exists** and Gazebo (`run_policy_episodes.py --weights`) **requires** `.npz`. Gazebo backend is unselectable and unusable from the UI. | server.py:467; `ls rl/runs/*.npz` | Part 6.5 `.npz` group + backend-aware options. |
| D-052 | No engine backend parameter at all: only VecSim (`rl/evaluate.py`) is wired. `simulation/gazebo/run_policy_episodes.py` (the L6 harness) is unreachable from the dashboard. | index.html:73-80; server.py:475-517 | Part 6.2 `backend: vecsim|gazebo_l6`. |
| D-053 | No `sensing` selector for L6 (`emulated|camera|camera_aruco|camera_dr`), no `--student-shield`, no `--save-traces` for gazebo, no target geometry. | run_policy_episodes.py:470-505 (args exist, unused) | Part 6.2 full option plumbing. |
| D-054 | `_leaderboard()` sets L7 field runs `level:"L_real"` and `track:"l7"` (server.py:211) → L7 card empty (D-017); L6 card empty (no `level:"L6"` exists); cards claim “best success per level” but two of eight cards can never populate. | server.py:204-231, app.js:108-113 | Part 5.2. |
| D-055 | `_track_for` lets `source` (`eval_json`,`archive`) fall through as `track`; UI then presents sources as tracks. | server.py:86-99; app.js:24 | Part 5.1. |
| D-056 | Contract `_norm_contract_checks` defaults missing `pass` to **True** (server.py:566) → skipped/missing checks render as green passes; SKIP entries are indistinguishable from real passes. | SS-09; server.py:561-586 | Part 6.3 tri-state checks, `pass=None` for skip. |
| D-057 | Contract `status` = `report.get("verdict") or ("PASS" if passed else "INFO")` (server.py:725); UI maps “not contains pass” → red fail banner (app.js:1066) → **INFO (a healthy no-verdict state) renders as an error**. | SS-09 | Part 6.3 banner mapping. |
| D-058 | `gazebo_l6_gate` check renders `null / N/A` with no explanation of what the gate is, how to satisfy it, or what command produces the evidence. | SS-09; audit_contract.py:~850 | Part 6.6 gate help text + command. |
| D-059 | `replay_list()` only picks gazebo `*.traces.json` (server.py:392-396) — the 5-ep/3-ep Gazebo JSONs (`student_L2_cons_adr_s0_gazebo_L1_5ep.json` etc.) are leaderboard rows but never replayable; Gazebo media route missing (D-047). | server.py:381-401 | Part 7.3 replay coverage + media route. |
| D-060 | `/api/overview` Pi bench path fallback chain is three speculative shapes deep with no schema — fragile but harmless; low priority. | server.py:73-83 | Leave as-is; note. |
| D-061 | `/api/rollout` doesn’t validate `policy` against the options list (arbitrary `.pt` path from client is passed through) — low risk (local tool) but preflight existence check fixes both this and D-049. | server.py:475-505 | Part 6.2. |
| D-062 | No job history endpoint; jobs vanish on restart; `obs_<ts>.json` orphans accumulate in `rl/runs/` with no GC or listing. | server.py:34, 475-517 | Part 6.4 job list (last 10, in-memory ok). |

### 1.5 Verified runtime facts (so you don’t re-derive them)

1. `rl/runs/student_L2_cons_adr_s0.pt` — exists, 1.29 MB. `rl/runs/student_L2_cons_adr_s0.npz` — exists. `evaluate.py:load_policy` does `torch.load(spec)`; spawned with `cwd=rl/`; spec is bare name → **FileNotFoundError, always**. Fix is path resolution, not a retrain.
2. Gazebo harness CLI (verified `run_policy_episodes.py:447-505`): `--weights <npz>`, `--bridge {fake,gazebo}`, `--sensing {emulated,camera,camera_aruco,camera_dr}`, `--episodes`, `--level`, `--seed`, `--save-traces PATH`, `--out`, `--json`, `--max-time 45`, `--marker-layout {real,legacy}`, `--lighting-dr`, `--card-tilt-dr`, `--smoke`, `--baseline`. It does **not** have a `--target` flag — STRL001 physics in Gazebo needs a world/obstacle change; see Part 7.5 for honest gating.
3. `rl/sim.py` has **no** STRL001 obstacle model → in VecSim, `target=strl001_belly` today affects rendering/camera-POV only. State this in the UI when selected (Part 7.5). Do not fake physics parity.
4. Gazebo L6 eval summaries exist with levels `L1`/`L2` (e.g. `student_L2_cons_adr_s0_gazebo_L1_5ep.json`, `success 0/5`); zero `*.traces.json` under `rl/runs/gazebo_l6/` yet → **no Gazebo replay bundles exist**; the tab’s Gazebo path is effectively untested end-to-end.
5. Tag constellation: real drone = ArUco **14/6/3**; STRL001 = **20/21/22**; legacy layout 9/12/13/14 appears in configs — the Bundle/geometry panels must state which layout is loaded.
6. `STRL001` numbers already in code (app.js:459-610): nose wheel `(0,-0.45) r0.075`, mains `(±0.28,-0.05) r0.085`, fin-tip skids `(±0.58, 0.34→0.48)`, pod `z≈0.40`, wing `z=0.585`, nacelles `±0.43 z=0.655`, chin turret `z=0.24 r=0.055`, belly tag `#22 z=0.345`, fin tags `#20/#21 z=0.245`. These are **unsurveyed** (`surveyed:false` per `docs/council_v2/03_strix_plane_geometry_model.md`) — render as “nominal, unsurveyed”, never as measured.

### 1.6 Screenshot → defect cross-reference

| Screenshot | Contents | Defects shown |
|---|---|---|
| SS-01 `...-5f570a85...png` | Level `<select>` open | D-001, D-002 (native popup) |
| SS-02 `...-51e85898...png` | Curves tab crop: legend soup + sidebar checkboxes | D-026, D-033, D-027 |
| SS-03 `...-5ab81dda...png` | Replay tab full: failed job, white Run, white scrollbar, Layers detached | D-019, D-021, D-002, D-034, D-042 |
| SS-04 `...-6ae36441...png` | huggingface.co Cosmos3 page | **Not the product.** Out of scope — do not import that aesthetic (see §2 note) |
| SS-05 `...-f0d16211...png` | Pills closeup row | D-015, D-016, D-024 |
| SS-06 `...-6d553340...png` | Leaderboard full @localhost:8768 | D-001..D-008, D-012, D-016, D-017, D-037, D-038 |
| SS-07 `...-89dceecf...png` | Layers sidebar closeup | D-002, D-034 |
| SS-08 `...-920f3e4b...png` | Training & ADR Curves full | D-026..D-032, D-039 |
| SS-09 `...-9810da75...png` | Sim-vs-Real Contract | D-006, D-007, D-056, D-057, D-058 |
| SS-10 `...-9c780e5a...png` | Bundle dropdown flat list | D-020, D-048, D-002 |
| SS-11 `...-e0cbbc0b...png` | Rollout Trigger policy dropdown | D-049, D-011, D-002 |
| SS-12 `...-f0d16211...png` (pills) | Filter pills closeup | D-016, D-035 |
| SS-00 `...-2feb11ba...png` | RealFlight 9.5 `STRL001` loaded (reference) | Target airframe ground truth; Appendix B |

---

## Part 2 — Clean, Daylight-First Academic/Engineering Research Theme

Design intent: **lab notebook printed on warm paper**, not IDE, not SaaS marketing. Crisp slate ink, one restrained accent, hairline borders, tabular figures, generous row height, zero gradients, zero glow. Dark mode remains as an optional toggle for the workshop at night; it is not the default and it is not a separate design.

Explicitly **do not** copy SS-04 (Hugging Face dark page): that is a marketing surface, not an instrument.

### 2.1 Tokens

```css
:root{
  color-scheme: light;
  /* paper + ink */
  --bg:        #f8fafc;   /* slate-50, app canvas */
  --surface:   #ffffff;   /* cards, panels */
  --surface-2: #f1f5f9;   /* slate-100, insets, table header, chips */
  --surface-3: #e2e8f0;   /* slate-200, pressed states */
  --border:    #e2e8f0;   /* hairline */
  --border-2:  #cbd5e1;   /* slate-300, strong hairline / inputs */
  --fg:        #0f172a;   /* slate-900, primary ink */
  --fg-2:      #334155;   /* slate-700, secondary ink */
  --muted:     #475569;   /* slate-600, captions — ≥7:1 on white */
  --faint:     #64748b;   /* slate-500, disabled/hints — ≥4.5:1 on white */
  /* single accent, engineering blue */
  --accent:    #1d4ed8;   /* blue-700, links/active — 6.4:1 on white */
  --accent-2:  #2563eb;   /* blue-600, hover/large fills */
  --accent-we: #eff6ff;   /* blue-50, selected bg */
  /* semantic (all with matching 50-tint bg + 200 border for chips) */
  --good:#15803d; --good-bg:#f0fdf4; --good-bd:#bbf7d0;  /* green-700/50/200 */
  --bad: #b91c1c; --bad-bg: #fef2f2; --bad-bd: #fecaca;  /* red-700/50/200 */
  --warn:#b45309; --warn-bg:#fffbeb; --warn-bd:#fde68a;  /* amber-700/50/200 */
  --info:#1d4ed8; --info-bg:#eff6ff; --info-bd:#bfdbfe;  /* blue-700/50/200 */
  --skip:#475569; --skip-bg:#f8fafc; --skip-bd:#e2e8f0;  /* neutral, not green */
  /* data-viz categorical (light-safe, colorblind-aware, fixed order) */
  --c1:#1d4ed8; --c2:#047857; --c3:#b45309; --c4:#7c3aed;
  --c5:#be123d; --c6:#0e7490; --c7:#4d7c0f; --c8:#9f1239;
  --thumb:#cbd5e1;        /* scrollbar thumb */
  --grid:#e2e8f0;         /* chart gridlines */
  --grid-2:#cbd5e1;       /* chart axis line */
  --ok-shadow: 0 1px 2px rgba(15,23,42,.06), 0 1px 3px rgba(15,23,42,.05);
  --r-sm:6px; --r-md:10px; --r-lg:14px;
  --ring: 0 0 0 2px var(--bg), 0 0 0 4px var(--accent-2);
}
[data-theme="dark"]{
  color-scheme: dark;
  --bg:#0f172a; --surface:#1e293b; --surface-2:#0b1220; --surface-3:#334155;
  --border:#334155; --border-2:#475569;
  --fg:#e2e8f0; --fg-2:#cbd5e1; --muted:#94a3b8; --faint:#64748b;
  --accent:#60a5fa; --accent-2:#3b82f6; --accent-we:#1e3a5f;
  --good:#4ade80; --good-bg:#052e16; --good-bd:#166534;
  --bad:#f87171;  --bad-bg:#450a0a;  --bad-bd:#7f1d1d;
  --warn:#fbbf24; --warn-bg:#422006; --warn-bd:#92400e;
  --info:#93c5fd; --info-bg:#172554; --info-bd:#1e40af;
  --skip:#94a3b8; --skip-bg:#1e293b; --skip-bd:#334155;
  --c1:#60a5fa; --c2:#34d399; --c3:#fbbf24; --c4:#a78bfa;
  --c5:#fb7185; --c6:#22d3ee; --c7:#a3e635; --c8:#f472b6;
  --thumb:#475569; --grid:#1e293b; --grid-2:#334155;
  --ok-shadow: 0 1px 2px rgba(0,0,0,.4);
}
```

Rules:
- Default render is **light**. First paint respects `prefers-color-scheme` only when user hasn’t chosen; choice persists in `localStorage["obs-theme"]`. Theme toggle is a sun/moon SVG icon-button, top-right of topbar (D-001).
- One accent only. No gradients, no glow, no neon. Success/failure are the semantic colors, used as **text + 50-tint chip + 200 hairline border**, never as full-card fills except the contract banner.
- Every color above is pre-checked ≥ 4.5:1 on its intended background (Part 8 has the checklist).

### 2.2 Typography

```css
--font-ui:   system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
--font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
body{ font: 13px/1.45 var(--font-ui); font-variant-numeric: tabular-nums; }
```

- Numbers in tables, pills, axis labels, tooltips: `font-variant-numeric: tabular-nums` + `--font-mono` where alignment matters (Run IDs, metrics, timestamps).
- Scale: 11px captions (`.cap`), 12px table/secondary, 13px body, 15px card emphasis, 20px/600 page title in topbar (`17px` if space-constrained). No other sizes.
- Headings are sentence case, weight 600, ink `--fg`. No all-caps except 10px/600 letterspaced micro-labels (`.klabel`, e.g. `LEVEL`, `TRACK`, `SOURCE` over filter clusters — 0.06em tracking, `--muted`).
- Units are always present and separated by a thin space in mono: `0.136 ms/step`, `171,916 steps/s`.

### 2.3 Layout & spacing

- 8px base grid. Card radius `--r-md`, border `1px solid var(--border)`, background `--surface`, shadow `--ok-shadow`. Section gaps 16px; card padding 12–16px.
- Topbar: 48px, `--surface`, hairline bottom border. Brand left (`UGV Research Observatory` + separator dot + dim subtitle `Mecanum autopark · ADR`), center pills (D-log counts), right: theme toggle + auto-refresh switch + `Updated 11:32:07` (D-014, D-024).
- Tabs row: 40px, `--surface`, hairline bottom border; tabs are text buttons with 2px underline indicator in `--accent` when active (not filled pills); count badges (e.g. `Leaderboard 77`) optional, muted mono.
- Every tab pane starts with a **one-line purpose statement** in `--muted` 12px (see Part 3.2). No pane opens mute.

### 2.4 Scrollbars (D-002)

```css
*{ scrollbar-width: thin; scrollbar-color: var(--thumb) transparent; }
*::-webkit-scrollbar{ width:10px; height:10px; }
*::-webkit-scrollbar-thumb{ background:var(--thumb); border-radius:6px;
                             border:2px solid transparent; background-clip:content-box; }
*::-webkit-scrollbar-track{ background:transparent; }
*::-webkit-scrollbar-thumb:hover{ background:var(--faint); background-clip:content-box; border:2px solid transparent; }
```
Apply everywhere including `body`. Native `<select>` popup scrollbars cannot be themed — accept OS styling there but constrain list length with `<optgroup>` (Part 7.2) so popups stay short.

### 2.5 Controls contract (D-003, D-008, D-033, D-034, D-044)

Every interactive element uses these classes; **no raw OS-native buttons anywhere except icon-only where unavoidable:**

- `.btn` — base: `--surface` bg, `1px var(--border-2)` border, `--fg` text, `--r-sm`, padding `6px 12px`, 12px/500, hover `--surface-2`, active `--surface-3`, `:focus-visible{ box-shadow: var(--ring) }`, `:disabled{ opacity:.45; cursor:not-allowed }`.
- `.btn-primary` — `--accent` bg, white text, hover `--accent-2`. (Run / Trigger button.)
- `.btn-ghost` — transparent border, `--muted` text, hover `--fg`. (Icon buttons, Close.)
- `.btn-danger` — `--bad` text, `--bad-bd` border, `--bad-bg` hover fill. (Stop/clear only — no destructive action exists today, reserve.)
- `.chip` — filter pill: `--surface-2` bg, `--border`, `--fg-2`, radius 999px, 12px, padding `3px 10px`; `.chip[aria-pressed="true"]` — `--accent-we` bg, `--accent` border+text, 1.5px inset ring. Counts inside chips optional mono muted.
- Checkbox / radio: `accent-color: var(--accent)`, `width:14px;height:14px;margin:0; flex:0 0 auto`, wrapped in flex rows with `gap:8px; min-height:28px` (Part 7.1). Never bare input + text node in an inline label.
- Select: `appearance:none`, background-image chevron SVG (`data:image/svg+xml`, stroke `--muted`), `padding-right:28px`, border `--border-2`, bg `--surface`, radius `--r-sm`, focus ring.
- Search: `--surface` bg, leading magnifier SVG absolutely positioned, `padding-left:28px`, radius 999px? No — `--r-sm`, 260–320px wide, Esc clears.
- Range (`#scrubber`): custom track `--surface-3` 4px, filled portion via `background: linear-gradient(...)` sized to value (set `--fill` from JS), thumb 16px white with `--border-2` ring; `:focus-visible` ring. Works `::-webkit-slider-thumb` + `::-moz-range-thumb`.
- Toggle (auto-refresh): switch 32×18, `--surface-3` off / `--accent` on, knob white; pure CSS+`[checked]`.
- Number inputs: same as text; hide spin buttons (`appearance:textfield`) — values are small (1–20).
- Focus: all `:focus-visible` → `box-shadow: var(--ring)`; never `outline:none` without replacement.

### 2.6 Badges (D-007)

`.badge` = inline-flex, 2px/6px padding, radius 4px, 10px/600 letterspaced 0.04em, uppercase only for source tags. Variants: `.pareto` green (`--good-*`), `.archive`/`.eval_json` slate, `.gazebo_l6` blue (`--info-*`), `.l7` amber (`--warn-*`), `.active` pulsing dot + green text (running), `.legacy` amber outline (legacy `.log`), `.modern` green outline (`metrics.jsonl`), `.unsurveyed` amber (STRL001). Each badge carries a 6px leading dot in the semantic color. Text contrast ≥ 4.5:1 on chip bg.

---

## Part 3 — Information architecture & scientific hierarchy

### 3.1 The one-line mental model (put this in the Help drawer, verbatim)

> **We train a mecanum car to park itself under a drone (and, later, a fixed-wing’s belly) by RL. It graduates sim ladder rungs, then Gazebo, then a real Raspberry Pi on the car. This dashboard shows whether it has earned the next rung.**

`sim success ≠ real success` — the only gate that counts is `rl/evaluate.py` (1000 eps, seed 11, Wilson CI) and, ultimately, L7 on the car.

### 3.2 Tabs: renamed, subtitled, ordered

| Tab id | Title (unchanged) | Subtitle line, always visible 12px `--muted` |
|---|---|---|
| leaderboard | **Leaderboard** | `Gate results — 1000 seeded episodes (seed 11), Wilson 95% CI. Best run per ladder rung.` |
| curves | **Training & ADR Curves** | `Optimisation telemetry from PPO/ADR. Loss ≠ success — check the Leaderboard for gates.` |
| replay | **2D Layout & Episode Replay** | `Frame-by-frame episode playback: truth vs tags vs estimator, on the true-scale layout.` |
| contract | **Sim-vs-Real Contract** | `Sensor & plant parity: does the car behave on the bench the way the sim thinks it does?` |

Sidebar headings on each pane get the same one-line muted captions (e.g. above Rollout Trigger: `Operator-run evaluation — S0 read-only otherwise`).

### 3.3 Global Help drawer (`?` icon, top-right; `Esc` closes; persists no state)

Sections (accordion, all default-collapsed except Ladder):
1. **The ladder** — table, below.
2. **Metrics glossary** — every column/chart label: `success_rate`, `wilson_lo/hi` (95% CI of the success proportion, n=1000), `collisions`, `timeouts`, `false_park`, `Med Time (s)`, `Pos p95 (mm)`, `Yaw p95 (deg)`, `pol_loss/val_loss`, `aux_loss/dyn_loss`, `std` (policy action stddev), `entropy`, `mean_return`, `sps` (sim steps/s), `dr_scale` (domain-randomisation curriculum scale 0→1), `anchor_frac`. Each: unit, “good direction”, current gate threshold.
3. **What counts as parked** (from `docs/RL_PARKING.md`): |dy|≤30 mm, |dx|≤50 mm, yaw ≤5°, corners in corridor, still 1.0 s, zero contact, false-park counted separately.
4. **Where the data lives** — `rl/runs/*.json` evals, `rl/runs/*.metrics.jsonl` training, `rl/runs/gazebo_l6/` L6, `calibration/runs/l7/` field runs, `archive.jsonl`. One line each; “never committed: weights, frames, logs”.
5. **Safety** — `S0` read-only here; rollouts run sim only; no motor commands exist in this surface.

### 3.4 The ladder (render as a horizontal stepper in Help, and as the card row caption on Leaderboard)

| Rung | Name | What it proves | Gate (abbrev.) |
|---|---|---|---|
| L0 | Aligned start | Posterior-mean student, perfect state, 0.75 m ahead | ≥99%, 0 collisions |
| L1 | Bearing + noise | ±30° bearing, ±60° yaw, tags visible, 80 ms nominal latency | ≥98%, ≤0.5% coll |
| L2 | Full posterior + DR | 60-draw posterior, yaw-couple/deadzone/lag/gain/sag DR | ≥95%, ≤1%, beats EKF+staged |
| L3 | Dropouts | 25–45% drops, outliers, 3 s blackouts, latency 80–200 ms | ≥95%, ≤1%, false-park ≤0.5% |
| L4 | Open 4×4 m | Any heading, drone may start unseen, random drone yaw | ≥90%, ≤1%, ≤45 s median |
| L5 | Full lab | 24×12 m + furniture, scene-bridge only | ≥85%, ≤1% |
| **L6** | **Gazebo sim-to-sim** | Native rollers, rendered camera, real ParkingEstimator + RelativeHeading | ≥90% of L3/L4 rate, ≥100 eps |
| **L7** | **Real Pi (this repo)** | Operator-authorised, car on the floor, drone powered off | Staged ≥8/10 L1 → ≥18/20 L4, 0 contacts |

`L_real` = L2-like config with real-tag-latency profile trained in VecSim; `L_chaos` = stress curriculum. Explain on hover of the card.
Tooltips: every tab, every column header (`title=` attr + Help drawer link), every gate % on cards (“best 1000-ep Wilson lower bound this rung”).

### 3.5 Header pills, specified (D-014, D-024)

Left→right: **brand** (`UGV Research Observatory` + dim `· Mecanum autopark`) | pills: `Evals <n>` (tooltip: “eval JSONs + archive lines; hover for archive/gz split”), `Training logs <n>` (tooltip: “*.metrics.jsonl + legacy *.log under rl/runs/”), `L6 Gazebo <n>` (new: count of `gazebo_l6/*.json`; tooltip explains L6), `L7 runs <n>` (tooltip: “calibration/runs/l7 directories with ticks.jsonl”), `Pi4B <x> ms/step` (tooltip: source file path, or `n/a — not benched`) | right: `Updated HH:MM:SS` (auto), auto-refresh **switch** with real state text (`Auto-refresh 5s` / `Paused`), theme toggle, Help `?`.

### 3.6 Error & empty states (D-038)

- `fetchJSON` keeps returning `null`, but every call site must render a panel-level inline message: red-tinted strip, `--bad-bg`, icon, `Could not reach /api/leaderboard — is the server running?` + `Retry` button. Never a silent `—`.
- Empty states: centred icon + one sentence + one action. Examples: curves → `No training runs yet — metrics.jsonl files land in rl/runs/ during PPO.` + link to PLAN doc path; leaderboard → `No gate results match these filters.` + `Clear filters`; replay → `No bundle selected.` + `Choose a bundle or run a rollout.`; contract → `No L7 ticks found under calibration/runs/l7/*/ticks.jsonl.`
- A rollout job failure is NOT an empty state — it gets the Part 6.3 error panel with stderr tail.

---

## Part 4 — Charts: legend, axes, and “SPS must not crush std”

### 4.1 Layout per chart card

```
┌────────────────────────────────────────────────────────┐
│ <span class=chart-title>Losses · pol/val</span>          │  ← HTML, above canvas
│ <span class=chart-unit>unit: nats · log-y toggle</span> │  ← HTML muted caption (optional)
│ ┌─ div.legend (flex, wrap, gap 6px) ────────────────┐ │
│ │ ─ student_L2…·pol_loss   ─ student_L2…·val_loss   │ │  ← HTML legend, ≤2 rows, overflow “+6 more”
│ └─────────────────────────────────────────────────────┘ │
│ ┌─ div.chart-wrap (position:relative) ───────────────┐ │
│ │   <canvas>                                          │ │
│ │   <div class=crosshair-line>                        │ │  ← CSS-drawn 1px dashed line, display toggled
│ │   <div class=chart-tooltip>   (absolute)           │ │  ← HTML tooltip, pointer-events:none
│ └─────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```
No text is drawn inside the canvas except axis tick numbers and axis title. Titles, unit captions, legends, tooltips are all HTML — one styling system, selectable text, accessible.

### 4.2 Legend (replaces D-026)

- `div.legend` is `display:flex; flex-wrap:wrap; gap:6px 10px; max-height:44px; overflow:hidden` with a `+N more` toggle button that expands to full height.
- Item: line swatch (2px line, 18px wide, CSS or inline SVG) in the series color + 10px label `run_id · metric` truncated to 34 chars with `title=full` (mono 11px) + hover: others dim to 0.25 opacity; click: toggle series visibility (state kept in `state.seriesHidden`, survives re-render); when hidden, swatch becomes hollow and label struck-through muted.
- 15 series × 3 runs = worst case: legend wraps to ≤3 rows or collapses behind `+N more`. **Text never overlaps.** Hard rule.
- Colors assigned per **run** (stable hash → palette index), metric distinguished by **dash**: `solid` = first metric of chart, `6 3` dash = second, `2 3` dot = third, `10 3 2 3` second-op for fourth. Document in legend swatch.
- Selecting >4 runs in Curves shows an inline amber warning chip in the sidebar: `>4 runs — curves are unreadable. Consider fewer.` Not a block.

### 4.3 Crosshair hover + tooltip (replaces D-030, D-031, D-032, D-040)

- One shared implementation in `attachChartHover`; `mousemove` → rAF-throttled → compute hovered iter via **binary search** on the union of visible series’ iter arrays (all series sorted by `iter` — assert on load; if unsorted, sort a copy).
- Vertical 1px dashed `--faint` line rendered as an absolutely-positioned `div` (no canvas redraw for the line); points of each visible series snap to nearest iter within `x` tolerance; snapped points get a 3px ring drawn on canvas.
- Tooltip `div.chart-tooltip`: `iter <n>` header, then one row per visible series: swatch, `run·metric`, value right-aligned mono (`toPrecision(4)` or engineering notation for |v|≥1e4 → `1.72e5`), or `—` when that series has no point at that iter. Rows sorted: same chart order. Positioned right of cursor, flips at canvas edge, never overflows card (clamp + `max-width:280px`).
- `mouseleave` clears line+tooltip. **No canvas redraw per mousemove except the snapped-point ring** — redraw only on hover index change, and only the ring layer if implemented as second canvas; acceptable: full redraw throttled to rAF, but cache the base render (draw to offscreen canvas once per data/hover-visibility change; blit on hover).
- Hit-testing uses `getBoundingClientRect`, not `c.width` — cursor coordinates are client px; canvas is DPR-scaled (see below).

### 4.4 Axes (replaces D-027)

- **One chart, one unit.** Y-axis label (top-left inside card, 11px `--muted`): e.g. `rate (0–1)`, `loss (nats)`, `steps/s`, `x return`. X-axis title bottom-right: `iter`.
- Left pad computed dynamically: `padL = max(44, ctx.measureText(widestTick).width + 18)`; right pad 12; top 8 (canvas starts below HTML legend); bottom 26.
- Ticks: 5 gridlines, `niceNum` algorithm (1/2/2.5/5 ×10^n). No negative zero. Format:
  - |v| ≥ 1e5 → `171.9k` or `1.72e5` (choose k-notation ≤ 1e6, e else)
  - 1e3–1e5 → `12.4k`
  - 1–1e3 → trim to 3 sig figs, strip trailing zeros (`0.05`, `0.25`, never `0.250`)
  - <1e-2 → `1.2e-3`
- Gridlines `--grid`, axis baseline `--grid-2`, tick text `--muted` mono 10px. Zero line (if visible) drawn `--border-2`, 1.5px.
- X ticks: 4–5 nice ticks + always first/last iter label. No rotated text.
- **SPS can never appear beside std again** (see 4.5). Mixed-unit overlays are a spec violation — a reviewer rejects the PR on sight.
- Y-axis per chart is shared across series in that chart (all same unit). No dual-axis charts in this app — instead, small multiples (4.5).
- Log-y toggle on the Losses chart only (`checkbox` in card header, persisted in localStorage). Log scale uses natural log gridlines with 10× tick labels (`1e-3, 1e-2, …`).

### 4.5 Corrected chart registry (replaces D-028, D-029, D-036)

`CHART_SPECS` v2 — each chart declares `unit`, `fmt`, `zeroBaseline`, `series`, and cards render in a responsive 2-col grid (1-col ≤1100px):

| # | Title | Series (metric keys) | Unit / scale | Notes |
|---|---|---|---|---|
| C1 | **Episode outcomes** | `succ_rate` (solid), `coll_rate` (dash), `timeout_rate` (dot), `false_park_rate` (dashdot) | rate 0–1, linear, zero-baseline | The chart everyone actually reads |
| C2 | **Curriculum** | `dr_scale` (solid), `anchor_frac` (dash) | fraction 0–1, linear | Own card; separate y from C1 |
| C3 | **Policy losses** | `pol_loss`, `val_loss` (per run) | nats, linear or log (toggle) | Per-run paired lines; ≤4 runs |
| C4 | **Aux & dynamics losses** | `aux_loss`, `dyn_loss` | nats, linear/log (follows C3 toggle) | `std` **removed** (was illegally mixed) |
| C5 | **Action stddev** | `std` per run | stddev (unitless) | Its own chart; axis auto 0.02–0.5 typ |
| C6 | **Entropy** | `entropy` per run | nats, auto-range | Was promised in title, never plotted |
| C7 | **Mean return** | `mean_return` per run | return units, auto | Was promised in title, never plotted |
| C8 | **Throughput** | `sps` per run | steps/s, k-format (`171.9k`) | Its own chart; never shares an axis |
| C9 | *(optional)* **Wall-clock** | `wall_s` | minutes | Collapsed by default (`<details>`) |

Titles must match contents — no exceptions. If a metric is absent from `metrics.jsonl`, the series simply doesn’t render and the legend shows the run with a dimmed `no entropy` note (parse once per run, cache).

Layout: `C1–C4` 2×2 grid; `C5–C8` 2×2 grid below; each card min-height 240px (canvas 180px). On ≤1100px, everything 1-col.

### 4.6–4.8 Implementation rules

- Canvas: `c.width = clientWidth*dpr; c.height = clientHeight*dpr; ctx.scale(dpr,dpr)`; `image-rendering: auto`; resize via `ResizeObserver` → re-render (debounced 100 ms). Font: `11px var(--font-mono)` — measure with the same font before setting padL.
- Offscreen base cache per chart: recompute on (data change | series toggle | hover visibility | resize); blit + ring on hover index change.
- Legend swatch in HTML uses inline style `border-top:2px solid/ dashed` per dash mapping — no canvas.
- Strips (D-043): give each strip an HTML header row above the canvas: `TAG PATH — valid / blackout / outlier + tag_age (s)` with right-aligned live readout (`t=12.34s · age 0.21s`), same for `GYRO — imu_w (rad/s), imu_yaw (pose)` and `ACTION — vx vy w (norm.) amber=|w|>0.8, red tick=shield blocked`. Add min/max captions in 10px muted mono at right. No text inside canvas except the playhead label removal (drop it — the scrubber shows t).
- Axis/legend text must not use canvas `fillText` at <10px; if it doesn’t fit, it’s HTML.

---

## Part 5 — Leaderboard fixes

### 5.1 Filters: three labeled clusters (D-016, D-035)

Replace the single `.filters` soup with a grouped toolbar, wrap-safe:

```
LEVEL   [ALL] [L0] [L1] [L2] [L3] [L4] [L5] [L_real] [L_chaos]        ← label = .klabel
TRACK   [ALL] [student] [teacher] [baseline]                          ← what the policy is
SOURCE  [ALL] [eval_json] [archive] [gazebo_l6] [l7]                  ← where the row came from
                                              [◯ Pareto only]  [Search run id…    ⏺]
```

- Each cluster is a `.filter-group` with a `.klabel` caption above or leading inside; groups separated by vertical hairline; `flex-wrap` per group (a wrapped row keeps its cluster intact — labels make ownership obvious).
- Drop `L6`/`L7` from LEVEL (D-017 — they never match). `gazebo_l6` and `l7` live in SOURCE. Pareto stays a standalone toggle. Search right-aligned, 260px.
- `aria-pressed` on chips; keyboard: `Tab` into group, arrows move between chips, `Space` toggles.
- Active filters echo as removable chips above the table (`Level: L1 ×` `Source: gazebo_l6 ×` `Pareto ×` `Clear all`) — one click to a clean board.

### 5.2 Level cards row (D-017)

Eight cards, same rungs, all populated or honestly empty:

`L0 · L1 · L2 · L3 · L4 · L5 · L_real · L_chaos` from evaluated entries (as today) **plus**:

- **L6 card** — populated from `track=='gazebo_l6'` rows: `<best success>% succ` + sublabel `gazebo · <n> eps` or `—` with caption `no Gazebo L6 runs yet` (honest empty, tooltip explains what Gazebo L6 is and links Help §Ladder).
- **L7 card** — latest `l7` run outcome (`parked`/`aborted`…) + duration, or `—` / caption `no field runs`.
- Card shows: rung id (`.klabel`), big `% succ` (24px/700 tabular), sublabel run_id truncated + `title`, and a 3px bottom bar whose width = success rate, color = green ≥ gate else amber/red vs the rung gate from 3.4. Failed gate → value in `--bad`. This makes the row a *gate dashboard*, not decoration.

### 5.3 Table

- Right-align numeric columns (`Success/Eps`, counts, times, p95); Run ID/Track/Level left; Badges left. Header labels keep `title=` with definitions from the glossary (3.3).
- **Zebra rows** `--surface-2` at 50%; hover `--accent-we`; row click = pin run to compare (alternative to Diff checkboxes; Diff button stays, now a `.btn` small).
- Wilson cell: keep the 60px bar, add point-estimate dot; CI text exact `[0.94, 0.99]` mono. If data missing: `—` + tooltip “no success/episode counts in source JSON”.
- YAW warning: rows with `Yaw p95 > 60°` get amber text on that cell (sanity flag), not row-wide color.
- Sort indicators `▲/▼` → SVG chevrons, `aria-sort` on `th`.
- Max height 480px, themed scrollbar, sticky header, `colspan` empty-state row inside tbody when filter yields 0 rows (`No matches — Clear filters`).

### 5.4 Diff panel (D-037)

Two-column metric diff: metric name (mono muted) | A | B; deltas column with sign-colored value (`+3.2%` green if better, red if worse — “better” defined per metric direction table in Help). Close button `.btn-ghost`. Header: `A vs B` with run ids, swap button (⇄) to flip columns.

### 5.5 L7 section (D-025)

`<details class=l7-section>` → styled summary row (chevron SVG, `Real Pi — L7 field runs`, count chip, hairline top border), table with columns `Run · Mode · Outcome · Duration · Hz · vision ms p50 · loop ms p50/p95 · End frame`. `End frame` column: `View` `.btn.small` when `has_end_jpg`, else muted `—`. Outcome cell chips: `parked` green, `collision` red, `timeout` amber, `aborted` slate.

### 5.6 Badges

`Pareto` green chip; source as slate/blue/amber chip per SOURCE group (`eval_json` slate, `archive` slate-outline, `gazebo_l6` blue, `l7` amber); `active` live dot for running jobs. No badge without a glossary entry.

---

## Part 6 — Rollout Trigger: rebuild (the tab’s broken heart)

### 6.1 Behaviour contract

The sidebar “Rollout Trigger” panel triggers **one operator-authorised sim evaluation** and auto-loads its replay bundle. It must be impossible to launch a silently-broken job. States: `idle → validating → running (elapsed s) → done | failed (reason + stderr tail)`. While running: form disabled, Cancel button (kills proc, marks job `cancelled`).

### 6.2 API changes (`server.py`)

`POST /api/rollout` v2 body:

```json
{ "backend": "vecsim" | "gazebo_l6",
  "policy": "<basename from /api/rollout/options>",
  "level": "L0..L5|L_real|L_chaos (L6/L7 rejected 400)",
  "episodes": 1..20, "seed": int,
  "sensing": "emulated|camera|camera_dr",      // gazebo_l6 only
  "target": "quadrotor_skids|strl001_belly",   // see Part 7.5 (gating)
  "student_shield": "none|sim|gazebo" }        // gazebo_l6 only, advanced, default none
```

Server-side, in order:
1. **Resolve** policy: lookup in the options set returned by `/api/rollout/options` (server builds it); reject anything else `400 {error:"unknown_policy"}`. Resolution to absolute path happens **here** (`RL_RUNS/<name>` for `.pt`, same for `.npz`; baselines pass through as the sentinel strings `oracle_pd`/`brake`).
2. **Existence preflight**: `Path.is_file()` → else `400 {error:"policy_not_found", path}`. This is the fix for the shipped `FileNotFoundError` (D-049).
3. **Backend preflight**: `vecsim` → check `rl/.venv/bin/python` (fallback `python3` with torch importable, else `400 {error:"vecsim_env_missing"}`). `gazebo_l6` → check `shutil.which("gz")` (or `gazebo`) plus `simulation/gazebo/run_policy_episodes.py` exists, plus **policy suffix must be `.npz`** (`400 {error:"npz_required", hint:"run rl/export.py"}`), plus `rl/runs/gazebo_l6/` writable. Fail fast with the code, before spawn.
4. Spawn with absolute paths. `cwd=REPO/rl` for vecsim (unchanged behaviour, now correct), `cwd=REPO/simulation/gazebo` for `gazebo_l6`.
   - `vecsim`: `[py, "evaluate.py", "--episodes", N, "--level", L, "--policy", ABS, "--seed", S, "--save-traces", trace, "--json", json]`
   - `gazebo_l6`: `[py, "run_policy_episodes.py", "--bridge","gazebo", "--weights", ABS, "--level", L, "--episodes", N, "--seed", S, "--sensing", sensing, "--student-shield", shield, "--save-traces", trace, "--out", RL_RUNS/"gazebo_l6", "--json", json]`
5. Capture **stdout+stderr** to `JOBS[id]["log"]` (bounded, last 16 KB ring). `_watch` on exit: `done` (rc 0, bundle_id = trace stem, bundle must exist → else `failed` with `error:"trace_missing"`) or `failed` with `error:f"exit {rc}"` + `stderr_tail` (last 8 lines).
6. `GET /api/rollout/{job_id}` returns `{job_id, status, bundle_id, error, stderr_tail, elapsed_s, log_path}`.
7. `GET /api/rollout/options` v2 → `{ "groups": [ {"label":"Exported students (.npz — Gazebo / Pi)", "items":[{name, mtime, run_id}]}, {"label":"Checkpoints (.pt — VecSim only)","items":[...]}, {"label":"Baselines","items":["oracle_pd","brake"]} ], "levels":[...], "sensing":[...], "backends":["vecsim","gazebo_l6"] }`. UI renders `<optgroup>`s; selecting a `.pt` while backend=gazebo disables Run with hint `Gazebo needs .npz — export with rl/export.py`.

### 6.3 Error surfacing (D-050, D-056, D-057)

Rollout panel error block (only when `failed|cancelled`):
```
✗ Rollout failed — policy_not_found
  rl/runs/student_L2_cons_adr_s0.pt not readable.
  $ stderr tail (last 8 lines):  …
  [Re-run]  [Copy command]  [Open log path]
```
Copy command copies the exact argv; log path shown read-only. `role="alert"`. Success: green strip `✓ 5 episodes recorded — bundle obs_179… loaded below.` Sim-vs-Real banner mapping fixed: `PASS|pass → --good`, `FAIL|fail → --bad`, anything else (incl. `INFO`) → neutral `--skip` chip with the verbatim status string — never red (D-057). Contract checks: tri-state — pass ✓ green outline, fail ✗ red, `SKIP`/missing data → `–` slate chip + tooltip `not evaluated: <reason>`; summary line `12 pass · 1 fail · 2 skip` (D-056).

### 6.4 Panel layout (replaces index.html form)

```
ROLLOUT TRIGGER — Operator-run evaluation (never actuates the car)   [job #1794673578 ✗]
Backend      (•) Fast VecSim (rl/sim.py · ~2 min/1000 ep)   ( ) Gazebo L6 (gz sim · slower, real camera)
Policy       [Exported students (.npz)        ▾]  ← grouped optgroups; search-as-you-type filter input above select
Level        [L1 ▾]   Episodes [5]   Seed [11]
Sensing      [Emulated ▾]   (camera DR only when backend=gazebo_l6; hidden for vecsim)
Shield       [none ▾]                      (gazebo only; advanced, collapsed by default)
Target       [Quadcopter skids ▾]           ← Part 7.5
             [ Run evaluation ▶ ]  (disabled while running; shows spinner + “running 34 s”)
Last job     ✗ failed · exit 1 · stderr tail ▸   [Re-run]
```
Job history: last 10 jobs listed with status dot + bundle link (`[rollout] obs_1794… · L1 · 5/5 ok`). All buttons are `.btn`/`.btn-primary` — no native whites.

### 6.5 VecSim vs Gazebo L6 — the selector, precisely

| Aspect | Fast VecSim | Gazebo L6 |
|---|---|---|
| Runner | `rl/evaluate.py` (torch, GPU/CPU) | `simulation/gazebo/run_policy_episodes.py --bridge gazebo` |
| Weights | `.pt` checkpoint (or `.npz` via export) | `.npz` only |
| Sensing | analytical sensor model (latency/dropout DR) | `emulated` / `camera` (render+ArUco PnP) / `camera_dr` (+photometric DR) |
| Speed | seconds | minutes–tens of minutes |
| Success means | L-gate evidence (Wilson) | sim-to-sim transfer check (L6 gate) |
| Replay bundle | `traces/*.json` → `[eval]` | `gazebo_l6/*.traces.json` → `[gazebo_l6]` |
| Use when | iterating, gates, sweeps | pre-L7 reality check, camera-path bugs |

Inline help line under the backend radios (1 sentence each, tooltip links Help §Ladder).

### 6.6 Gazebo L6 gate surfacing in Contract tab (D-058)

`gazebo_l6_gate` check card: name, `status` chip, condition caption `≥1 Gazebo L6 run with success > 0 on the sha-pinned .npz for this policy`, and — when `SKIP`/ineligible — the exact reproduction command, copyable:

```
cd simulation/gazebo && python run_policy_episodes.py --bridge gazebo \
  --weights ../rl/runs/student_L2_cons_adr_s0.npz --level L1 --episodes 20 \
  --seed 11 --sensing camera --save-traces ../rl/runs/gazebo_l6/<policy>_L1.traces.json
```

plus `sha mismatch` detail when the L6 npz sha ≠ current bench weights sha (that is the whole point of `meta_sha` in the npz — say so in the tooltip).

---

## Part 7 — Sidebar discipline, bundle list, and first-class STRL001 / Gazebo L6

### 7.1 All checkbox rows (Curves run selector, Layers, L7 filter, auto-refresh in topbar) — D-033, D-034

```css
.checkrow{ display:flex; align-items:center; gap:8px; min-height:28px;
           padding:2px 8px; border-radius:6px; cursor:pointer; }
.checkrow:hover{ background:var(--surface-2); }
.checkrow input{ flex:0 0 auto; width:14px; height:14px; margin:0; accent-color:var(--accent); }
.checkrow .grow{ flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis;
                 white-space:nowrap; font-family:var(--font-mono); font-size:11px; }
.checkrow .tail{ flex:0 0 auto; }  /* right-aligned badge, e.g. (legacy .log) / (active) */
.checkrow input:checked ~ .grow{ color:var(--fg); font-weight:600; }
.checkrow input:focus-visible{ box-shadow: var(--ring); }
```
Run selector additionally: header row `[Filter runs…]` + `All · None` text buttons; group headings `metrics.jsonl (n)` / `legacy .log (n)` as sticky `.klabel` rows; the `(legacy .log)` tag becomes a right-side amber `.badge.legacy` — never inline after wrapped text. Row `title` attr = full id. Search filters by substring.

### 7.2 Bundle dropdown (D-020, D-048)

`<optgroup label="Gazebo L6">` / `<optgroup label="L7 — Real car">` / `<optgroup label="VecSim eval">` / `<optgroup label="Rollouts (this machine)">` with option text `[gazebo_l6] student_L2_cons_adr_s0_L1_5ep · L1 · 0/5` (outcome when known from summary) — the dropdown order: L6 → L7 → rollouts → eval, newest first inside each group. Plus a filter input above it (`Filter bundles…`) implementing type-ahead over option text (native `<select>` search is OS-dependent; do a custom filter by hiding non-matching options via `option.hidden = true` — safe and dependency-free). Raw `obs_<ts>` ids display as `[rollout] obs_1794673578 · failed/ok · <level>`.

### 7.3 Gazebo L6 replay coverage (D-047, D-059)

- Server: add safe media route `GET /api/replay/media/gazebo_l6/{run_id}/{rel:path}` serving files under `rl/runs/gazebo_l6/<run_id>_frames/` with the same `safe_path` guard + `..` rejection + content-type allowlist (`image/jpeg|png`). 404 otherwise.
- `replay_list()` includes gazebo **summary-only** JSONs **flagged** `{"has_frames": false}` so they appear in the bundle list with a `[no camera frames]` tail note — visible but honest; selecting them yields the empty-state banner in the camera pane, not a broken img.
- Camera POV pane empty states: `frames.length===0` for gazebo bundle → centred placeholder `No camera frames were recorded for this run — re-run with --sensing camera_dr to capture them.` + the Part 6.6 command. Real L7 frame loads via existing `/api/l7/...` route — unchanged.
- Badge precedence in POV header: `REAL L7 CAMERA` (amber) > `GAZEBO CAMERA` (blue) > `SYNTHETIC POV` (slate) > `end.jpg` button only when `end_frame_url` set (as today).

### 7.4 STRL001 Plane-Belly as a first-class target (D-022; Part 1.5 §1, §6; docs/council_v2/03 & 14)

Everything STRL001 lives in a labelled sidebar group (not scattered controls):

```
TARGET AIRFRAME
( ) Quadcopter skids   tags 14/6/3 · gear @ ±0.157 m
( ) STRL001 belly      twin-tractor · V-tail · tags 20/21/22   [UNSURvEYED]
    nose wheel (0, −0.45) · mains (±0.28, −0.05) · fin skids (±0.58, +0.41 avg)
    belly tag #22 z=0.345 · pod z=0.40 · wing z=0.585 · nacelles ±0.43 z=0.655
    [collapsible: full dimension table vs RealFlight 9.5 reference — docs/council_v2/strix_plane_realflight.png]
[ ] 2.5D Oblique view          (auto-enabled when STRL001 selected)
Geometry source: nominal — NOT surveyed. Check before trusting clearances.
```

- Selecting `STRL001 belly`: auto-checks 2.5D, sets `state.targetGeometry='strl001_belly'` (already implemented in app.js:358-361 — keep), re-renders, and flips the goal/blind-ring copy to belly semantics. The panel shows the `.badge.unsurveyed` amber chip + last-updated date of the dimension table.
- Rollout Trigger honours the same selector (Part 6.4 `Target`) — one source of truth, `state.targetGeometry`.
- STRL001 canvas layers (wing checkerboard, nacelles, fin checker accents, tags 20/21/22, PORT/STBD labels — app.js:459-610) are kept; only fixing: label collision pass, `UNSURvEYED` watermark chip drawn bottom-left when STRL001 active, and dimension labels hidden behind a `Geometry` toggle (default off) so the hero view stays clean.
- RealFlight reference image (`docs/council_v2/strix_plane_realflight.png`, mirrored in `docs/ui_critique/`) is linked from the dimension table header — `open reference` opens in new tab.

### 7.5 Honest physics gating (do not lie to the operator)

`target` affects what VecSim/Gazebo actually do — state it inline:

| backend × target | What actually happens today | UI behaviour |
|---|---|---|
| vecsim + skids | Full eval semantics | Enabled |
| vecsim + strl001_belly | **Same physics; geometry/camera-POV differ only** — VecSim has no belly obstacle model | Enabled with inline note: `Physics unchanged — target changes rendering & camera POV only` |
| gazebo + skids | Full L6 gate semantics | Enabled |
| gazebo + strl001_belly | Requires a STRL001 world SDF (not built) | **Disabled** with note: `No STRL001 Gazebo world yet — see docs/council_v2/03` — enabled the moment `simulation/gazebo/*strl*.sdf` exists or `--world` override lands |

The dashboard never shows a control that silently no-ops. If it isn’t built, it is visibly `not built` with a pointer to the doc.

### 7.6 L6 in the Leaderboard & header

- New SOURCE pill `gazebo_l6` (already in data) and `l7`; L6/L7 cards populate per Part 5.2; header gains `L6 Gazebo <n>` pill (Part 3.5); a persistent one-time coachmark on first visit (`localStorage` gate): `L6 = Gazebo sim-to-sim. Click the L6 card for how it’s run →` (dismissible).
- `replay_list()` shows gazebo bundles with outcome summary inline (`0/5 · L1`) so a failed L6 is obvious before opening.

---

## Part 8 — Accessibility & polish checklist

- Contrast (light): `--fg` on `--bg` 17.3, `--muted` on white 7.6, `--accent` on white 6.4, `--good`/--bad`/`--warn` on their 50-tints ≥ 4.5. (dark) same roles ≥ 4.5. Verify with a checker before merge; paste screenshots into PR.
- `:focus-visible` ring on every interactive element (`--ring`); tab order: topbar → tabs → pane controls → main content; skip-link `Skip to content` as first focusable element.
- `aria-pressed` on chips/toggles; `aria-sort` on sortable th; `role=status` on rollout job line; `role=alert` on error panel; `aria-label` on icon buttons; decorative SVGs `aria-hidden`.
- Canvas charts get `role="img"` + `aria-label` summarising series count and range; the HTML legend provides the same info non-visually.
- `prefers-reduced-motion: reduce` → disable auto-scroll animations, pulsing `active` dot, playback smooth-advance (step discretely).
- No emoji in UI; icons are inline SVG (feather-style), stroke `currentColor`.
- All numbers tabular; all mono content `user-select:text`; table cells `overflow:hidden;text-overflow:ellipsis` with `title`.
- 1024px min supported width; below that panes stack; topbar wraps; no horizontal page scroll ever (panes scroll internally).
- No console errors on any tab on load and after every action in §9.3.

---

## Part 9 — Build order & acceptance

### 9.1 Order (each phase independently shippable)

1. **P1 Theme & controls** — Part 2 tokens, scrollbars, controls contract, badges, theme toggle + localStorage. Fixes D-001..D-008, D-011..D-013 (partial). Delete nothing user-visible yet.
2. **P2 Charts** — Part 4 registry v2 + C1–C8, HTML legend, crosshair, DPR, y-format, clip x-axis, `<details>` C9 collapsed. Fixes D-026..D-032, D-036, D-039, D-040.
   - Delete `chart1..chart4` canvases; add C1–C9. No dangling references.
3. **P3 Leaderboard** — Part 5 (clusters, cards L6/L7, zebra, diff panel, badges, empty states). Fixes D-016, D-017, D-035, D-037, D-012.
4. **P4 Rollout rebuild** — Part 6 (server changes + panel rebuild + error panel + options optgroups + stderr capture). Fixes D-049..D-053, D-056..D-058, D-061, D-062.
5. **P5 IA & help** — Part 3 (subtitles, Help drawer, glossary, ladder stepper, header pills + Updated, help tooltips, empty/error states). Fixes D-014, D-015, D-018, D-023, D-024, D-025, D-038, D-045, D-046.
6. **P6 Sidebar & STRL001** — Part 7 (checkrow spec, run selector groups, bundle optgroups + filter, gazebo media route, camera POV empty states, STRL001 panel + gating). Fixes D-014, D-020, D-022, D-033, D-034, D-041, D-042, D-043, D-047, D-048, D-059.
7. **P7 Polish** — Part 8 audit, 1024/1280/1440/1920 pass, console clean, `prefers-reduced-motion`.

### 9.2 Hard requirements (reject the PR otherwise)

- `#` legend text never overlaps at any window width ≥ 1024px; verified with 1, 3 and 8 visible series.
- `sps` appears on **no chart** with any other metric; `std` appears only on its own card; `entropy` and `mean_return` charts exist and render real data from `metrics.jsonl`.
- `Run evaluation` with `student_L2_cons_adr_s0.pt` (vecsim) either completes or shows `policy_not_found` with the absolute path — **never** a bare `exit 1`.
- Gazebo backend refuses `.pt` pre-spawn with the `.npz` hint; with `.npz` it spawns `run_policy_episodes.py` and the job appears with `[gazebo_l6]` badge on completion.
- Contract `INFO` banner renders neutral slate; contract shows `N pass · N fail · N skip`; missing `gazebo_l6` gate shows the copyable command.
- L6 card populates from `rl/runs/gazebo_l6/*.json` (≥1 row today: 0/20 latw5, 0/5 cons_adr) — it must show real numbers, not `—`.
- Every checkbox sits on the same baseline as its label text (Part 7.1) at 220px sidebar width; nothing wraps to the next line except by explicit row design.
- Daylight theme is the default on first visit; toggle persists; all badges/chips/banners meet 4.5:1.

### 9.3 Verification script (manual, 15 min)

1. Fresh load (no localStorage) → **light** theme, no white buttons/checkboxes, thin gray scrollbars, `Updated` pill ticking.
2. `Ctrl/Cmd-K`? No — keep simple: click through all 4 tabs; each shows its subtitle; Help `?` opens drawer; `Esc` closes.
3. Leaderboard: switch LEVEL=L1, TRACK=student, SOURCE=gazebo_l6 → rows filter; `Clear all` restores; L6 card shows a percentage; click `Diff` on two rows → panel renders with units; `×` closes.
4. Curves: select 1 run, then 3 runs of mixed formats; legend wraps ≤ 3 rows, no overlap; hide a series via legend click; hover → crosshair + tooltip shows `iter`, sorted values, correct-per-series iter matching; yank window to 1024px — no page scrollbar; SPS chart readable (`171.9k`).
5. Replay: pick `student_L2_cons_adr_s0 …` from optgrouped list; play; scrub; toggle layers — checkboxes aligned; toggle STRL001 → 2.5D auto-on, unsurveyed chip visible, PORT/STBD labels render; strips show captions + live readouts; `end.jpg` button hidden for non-L7; Gazebo bundle without frames shows the honest empty state, not a broken image.
6. Rollout: run vecsim eval on `student_L2_cons_adr_s0.pt`, 2 episodes → completes and auto-loads a `[rollout]` bundle; then select `.pt` with backend=Gazebo → Run disabled with hint; flip backend → `.npz` warning gone.
7. Contract: banner neutral; checks tri-state; `gazebo_l6_gate` shows command; skips render `–`, not green.
8. Console: zero errors, zero 404s (except favicon if not set — set it).
9. Toggle dark theme; repeat 1–2 quickly; toggle back; reload → persisted.
10. `uv run ruff check tools/observatory && uv run pytest -q` (root suite) still green.

### 9.4 Out of scope / do not do

- No framework rewrite, no build pipeline, no CDN assets, no web fonts downloads (system fonts carry the daylight theme fine).
- No changes to `rl/`, gates, configs, or data files. No invented metrics. If a series is missing in the data, the UI says `no entropy`, it never fabricates.
- No deletion of the canvas replay engine — it is correct enough; only its text/label placement and DPR handling change.
- Do not “fix” `rl/sim.py` to add STRL001 physics — that is a training-stack decision, explicitly gated.
- Do not hide the `L_chaos`/`L_real` rungs; they confuse less with a tooltip than without.

---

## Appendix A — Screenshot evidence index

| File (`docs/ui_critique/`) | Depicts |
|---|---|
| `199ad92a-…-2feb11ba-a793-4498-9504-1503c042f077.png` | RealFlight 9.5 “Naval Air Station”, STRL001 loaded, `~CS_LMVS`/`~CS_Pivot1` missing-frame warnings, `Loaded: STRL001` — ground truth for the belly target |
| `199ad92a-…-51e85898-4242-499d-84e8-bb408709170d.png` | Curves tab crop — legend collision + sidebar checkbox detachment |
| `199ad92a-…-5ab81dda-5d1c-42c3-9475-f3cfb2ae7844.png` | Replay tab — failed job, native white controls, Layers detached, `No frames` panes |
| `199ad92a-…-5f570a85-30df-43c3-be57-84ae2170cc7b.png` | Dark native Level dropdown |
| `199ad92a-…-6ae36441-58cb-4a9e-8737-26dbb2a471a8.png` | huggingface.co Cosmos3-examples — **out of scope reference; do not imitate** |
| `199ad92a-…-6d553340-333b-4596-a313-591d6d6da495.png` | Leaderboard full frame, localhost:8768 |
| `199ad92a-…-89dceecf-b336-4408-ab6a-529f1bd0b4ce.png` | Layers sidebar closeup |
| `199ad92a-…-920f3e4b-4c2a-41cc-9352-6f6158048f34.png` | Training & ADR Curves full frame |
| `199ad92a-…-9810da75-6716-4479-b7c8-2c1d9895035e.png` | Sim-vs-Real Contract full frame |
| `199ad92a-…-9c780e5a-38d4-48e6-81c0-ac2719c8e444.png` | Bundle dropdown flat list closeup |
| `199ad92a-…-e0cbbc0b-411b-49d9-890d-f36b4c3ca579.png` | Rollout Trigger policy dropdown closeup |
| `199ad92a-…-f0d16211-1f8d-4037-a18f-d298728653fd.png` | Filter pill row closeup |

## Appendix B — Runtime root causes (verified by running the code path mentally against the tree)

1. **Rollout `FileNotFoundError`**: `server.py:489-505` spawns `REPO/rl/.venv/bin/python evaluate.py` with `cwd=REPO/rl` and `--policy <p.name>` where `p.name` comes from `RL_RUNS.glob("*.pt")` (`server.py:467`). `evaluate.py:35` does `torch.load(spec)` with no path join → resolves in `REPO/rl/` → `student_L2_cons_adr_s0.pt` does not exist there (it lives in `rl/runs/`) → exception → traceback → rc≠0 → `_watch` sets `failed`, `error="exit 1"`, UI prints `job job_… : failed`. Fix: §6.2 (resolve + preflight + stderr capture). Not a data problem; the file is 1.29 MB and present.
2. **Legend garble**: `renderChart` writes legend swatches at fixed `y=4`, advancing only `lx`; on `lx>W-80` it resets `lx=padL+4` but keeps `y=4` (`app.js:267-273`) → collisions; names average ~28 chars → 15 entries × ~190px each ≈ 2850px of text in an ~800px card. Fix: §4.2 HTML legend outside canvas.
3. **SPS crush**: single linear auto-range per chart (`app.js:242-249`), `chart4` mixes `std` (≈0.02–0.4) and `sps` (≈4e4–1.72e5) (`app.js:210`). Fix: §4.5 C5/C8 split.
4. **INFO red banner**: `app.js:1066` `(status.toLowerCase().includes('pass'))?'pass':'fail'` + server `status="INFO"` when no verdict (`server.py:725`). Fix: §6.3 tri-state mapping.
5. **Empty L6/L7 cards**: server hard-codes L7 field runs `level:"L_real"` (`server.py:211`); Gazebo evals keep their true level (`L1/L2`) so no row ever carries `L6`/`L7`; `LEVELS` includes `L6` with zero matches. Fix: §5.2.
6. **`.diff-btn` unstyled**: no rule exists in `style.css` (grep confirmed) → OS-native white button with black text. Fix: §2.5 `.btn`.
7. **STRL001 exists but is decorative**: geometry constants, canvas painters, POV renderer and a `#target-geometry` select are all wired (`app.js:358-361, 459-610, 897-941`) yet rollout can’t target it, vector physics ignores it, and its dims are unsurveyed. Fix: §7.4/7.5 — surface it, label it, gate it, never fake it.
8. **Gazebo replay path is empty today**: zero `*.traces.json` under `rl/runs/gazebo_l6/`; `replay_list()` would show `[gazebo_l6]` only for traces; camera media route absent → even a perfect Gazebo run yields a dead POV pane. Fix: §7.3.

---

*End of spec. Questions to the Observatory lead before P4 — the rollout path touches operator-triggered compute; do not half-ship §6.*
