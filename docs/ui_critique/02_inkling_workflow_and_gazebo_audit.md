# UI/UX Critique 02 — Robotics Workflow, Gazebo L6 Integration & Backend Bug Hunt

**Auditor:** UI/UX Research Critic 02
**Date:** 2026-11-14
**Scope:** `tools/observatory/` (backend `server.py`, `data.py`, `replay.py`, frontend `static/app.js`, `index.html`, `style.css`) against the 12 screenshots in `docs/ui_critique/`.
**Constraint honored:** `tools/observatory/` was **not** edited. Every fix below is a proposal with exact code; nothing in the repo was modified by this audit.

---

## 0. Evidence inventory

| Screenshot (`docs/ui_critique/199ad92a-…`) | Content | Auditable? |
|---|---|---|
| `2feb11ba….png` | RealFlight 9.5 "Unexpected error" dialog | **No.** Unrelated desktop capture (RC flight sim). Not the Observatory. Do not file Observatory work from it. |
| `6ae36441….png` | huggingface.co Cosmos3-Examples page | **No.** Unrelated browsing capture. |
| `5ab81dda….png` | **2D Layout & Episode Replay** — `job job_1794673578 : failed`, canvas "No pose in this frame", strips "No frames" ×3, transport `t=0.00s step 0` | Yes — §2, §3 |
| `5f570a85….png` | Rollout Trigger **Level** dropdown: L0…L5, L_real, L_chaos (no L6, no gazebo option) | Yes — §2.5 |
| `e0cbbc0b….png` | Rollout Trigger **Policy** dropdown: bare `.pt` filenames | Yes — §2.1 |
| `6d553340….png` | Leaderboard full view: L0 100% / L1 97% / L2 54% / L3 22% / L_real 32% / L_chaos 0% / **L6 — / L7 —**; chips incl. `L6`, `gazebo_l6`, `l7` | Yes — §4 |
| `f0d16211….png` | Leaderboard chip row close-up: `L6` chip placed after `L_chaos`, no `L7` chip | Yes — §4 |
| `9c780e5a….png` | Bundle dropdown (`student_L2_cons_adr_s0_L2_cons` highlighted; `obs_*` and `l7:*` mixed in); episode rows `#5 timeout dur=44.98 err=373mm` | Yes — §3 |
| `89dceecf….png` | Layers panel; transport `step 0/479` (a bundle that *did* load) | Yes — §3 |
| `920f3e4b….png` / `51e85898….png` | Training & ADR Curves: **legends wrapped onto the same line and overdrawn**, unreadable | Yes — §5.1 |
| `9810da75….png` | Sim-vs-Real Contract: red **INFO** banner, `valid_pose_rate 0`, `raw_tag_rate 0` for `20261113-181848-watch-shutter4ms` | Yes — §5.3 |

**Runtime checks executed (read-only) to confirm the diagnosis** — `uv run python` against the repo:

- `_counts()` → `evals 77, archive 44, logs 39, metrics 5, gazebo_l6 14, l7_runs 25, traces 11`.
  - `gazebo_l6` glob matches **14 entries = 13 files + 1 directory** (`baseline_L1_20ep_s11.json` is a *directory*), and one match (`parallel_backlog_plan.json`) is a planning list, not a run. `server.py:50-53` never checks `is_file()`.
- `_leaderboard()` at runtime (see §4): 12 `source=gazebo_l6` entries; **only the 6 whose run_id does not contain "student" got `track=gazebo_l6`**; the other 6 (all `student_*` gazebo runs, incl. `student_L2_cons_adr_s0_gazebo_L1_5ep.json`) got `track=student`. None got `level` other than `L1`/`L2`. All 25 L7 entries get `level="L_real"`, `track="l7"`, `episodes=1`.
- `rl/evaluate.py:load_policy()` (`spec in BASELINES` else `torch.load(spec)`) confirms a bare filename resolves against **cwd**, which `/api/rollout` sets to `REPO/"rl"` (`server.py:505`). Neither `.pt` is at `rl/` root — both live in `rl/runs/`.
- `traces/obs_1794673578.json` and `obs_1794673578.json` **do not exist** — the failed job wrote nothing, which is what the frontend auto-loads on failure (see §3.1).
- `rl/runs/student_L2_cons_adr_s0.npz` **exists** — the Gazebo runner consumes this, not the `.pt`.
- `simulation/gazebo/run_policy_episodes.py --bridge gazebo` exists and works standalone (`--weights`, `--episodes`, `--level`, `--seed`, `--sensing`, `--save-traces`) — the dashboard simply has no path to it.

---

## 2. Bug #1 — `job job_1794673578 : failed` after clicking **Run** (Rollout Trigger)

### 2.1 Root cause: checkpoint path is resolved against the wrong cwd

`tools/observatory/server.py:475-505`:

```python
cmd = [str(py), "evaluate.py", "--episodes", str(episodes), "--level", level,
       "--policy", policy, "--seed", str(seed), "--save-traces", str(trace_path), "--json", str(json_path)]
proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(REPO / "rl"))
```

`rl/evaluate.py:32-35` does `torch.load(spec)` for any non-baseline spec, with `cwd = <repo>/rl`.
`/api/rollout/options` (`server.py:465-472`) lists `p.name` — bare filenames like `student_L2_cons_adr_s0.pt` — and the browser posts that verbatim. The real files are at `rl/runs/student_L2_cons_adr_s0.pt` and `rl/runs/teacher_L0_horizon.pt` (both verified). So:

```
torch.load("student_L2_cons_adr_s0.pt")  →  FileNotFoundError: [Errno 2] No such file or directory: 'student_L2_cons_adr_s0.pt'
```

→ traceback → non-zero exit → `_watch()` (`server.py:520-529`) sets `status="failed"`, `error="exit 1"`. This is deterministic for **every** checkpoint in the dropdown, not just `student_L2_cons_adr_s0.pt`. The rollout feature has never worked from the dashboard.

**Fix A (minimal, server-side path fix):**

```python
# server.py, rollout() — resolve advertised policy ids to real absolute paths,
# and never let the client name an arbitrary path.
policy_id = req.get("policy", "oracle_pd")
if policy_id in BASELINES:                      # oracle_pd / brake
    policy_arg = policy_id
else:
    cand = (RL_RUNS / Path(policy_id).name)     # RL_RUNS = rl/runs
    if cand.suffix != ".pt" or not cand.is_file():
        raise HTTPException(400, f"unknown policy: {policy_id!r}")
    policy_arg = str(cand)                      # absolute path, cwd-independent
```

(or keep `cwd=REPO/"rl"` and pass `"runs/" + policy_id` — the absolute-path form is strictly more robust).

### 2.2 Aggravating factor: the failure is undiagnosable from the UI, and it detonates the Replay tab

`_watch()` discards the child's stdout/stderr entirely; the job only reports `error="exit <n>"` (`server.py:520-529`, returned by `rollout_status` `server.py:531-542`). The user sees the literal string `job job_1794673578 : failed` with no reason. Worse, the frontend (`app.js:1094-1106`) **auto-loads the bundle even on failure**:

```js
if(s?.status==='completed'||s?.status==='done'||s?.status==='failed'){
  clearInterval(poll);
  if(s?.bundle_id) loadBundle(s.bundle_id);   // ← runs on FAILED jobs too
}
```

`bundle_id` is set to `obs_1794673578` at job creation (`server.py:482,510`) and is **never cleared** on failure. So after the crash, the browser GETs `/api/replay/bundle/obs_1794673578` → nothing exists on disk (verified) → HTTP 404 → `fetchJSON` returns `null` (`app.js:36`, which swallows all errors) → `state.episodes=[]` → the Replay tab renders "No pose in this frame" + "No frames" ×3 (screenshot `5ab81dda`), while the bundle dropdown still shows the previously selected `student_L2_cons_adr_s0_L2_cons` (truncated `student_L2_cons_adr_s0_L2_cc…` in the screenshot) because `sel.value=id` only applies when an `<option>` with that value exists (`app.js:362`) — i.e., **the dropdown lies about what is on screen**.

**Fix B (capture and surface stderr; don't auto-load missing bundles):**

```python
# server.py — capture output to a per-job log the UI can fetch.
log_path = RL_RUNS / "traces" / f"{bundle_id}.job.log"
with log_path.open("wb") as lf:
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(REPO / "rl"),
                                                stdout=lf, stderr=lf)
JOBS[job_id] = {..., "log_path": str(log_path)}
# rollout_status: return {"status", "error", "log_tail": <last ~4 KB of log>} on failure.
```

```js
// app.js — rollout submit handler
if(s?.status==='done'){ loadBundle(s.bundle_id); }
else if(s?.status==='failed'){
  $('#rollout-status').textContent = `job ${jobId} failed: ${s.error||''} ${s.log_tail?('— '+s.log_tail):''}`;
  loadReplayList();          // keep showing real bundles; do NOT loadBundle() a nonexistent one
}
```

Also: `int(req.get("episodes"))` / `int(req.get("seed"))` (`server.py:479-480`) raise an unhandled `ValueError` → HTTP 500 on a non-numeric seed/episodes. Use the same clamping with `try/except` and return 422.

### 2.3 Why no Gazebo L6 rollout can be triggered — there is no code path at all

The dashboard can only ever run `rl/evaluate.py` (the fast fake sim). Concretely:

1. **No bridge concept anywhere.** `rollout()` (`server.py:475`) hardcodes `evaluate.py`. There is no `bridge` field in the request, no `simulation/gazebo/run_policy_episodes.py` invocation, and no UI control for it (`index.html:73-81` — form has only Policy/Level/Episodes/Seed).
2. **The Level dropdown cannot express it.** `/api/rollout/options` returns `LEVELS` (`data.py:12`: `L0…L5, L_real, L_chaos`) — verified in screenshot `5f570a85….png`. No `L6`. And `L6` is not a sim level in `rl/sim.py` anyway; it is the *campaign name* for "Gazebo closed-loop runs" (`rl/runs/gazebo_l6/`). The correct knob is `--bridge {fake,gazebo}` (+ `--sensing {emulated,camera,camera_dr}`), not a new level.
3. **The policy types don't match.** The Gazebo runner consumes an **exported NumPy `.npz`** (`--weights rl/runs/student_L2_cons_adr_s0.npz` — its `--help` text: *".npz student from rl/export.py"*, verified file exists at `rl/runs/student_L2_cons_adr_s0.npz`), and every Gazebo run JSON records `"weights": "rl/runs/student_L2_cons_adr_s0.npz"`. The dashboard lists only `*.pt` checkpoints (`server.py:467`). Even if the form were naively wired to the Gazebo runner, passing `student_L2_cons_adr_s0.pt` → `StudentPolicy.load(... .pt)` → failure (export format mismatch). A dashboard Gazebo path must map `*.pt → rl/export.py → sibling .npz`, or list/verify `.npz` files.
4. **Different runtime profile.** `--bridge gazebo` launches Gazebo Harmonic, needs a display/GPU, ~45 s/episode wall time (the shipped runs record `wall_s: 1210` for 100 episodes), and writes into `rl/runs/gazebo_l6/`. It needs the same job machinery as `/api/rollout` (async watch, log capture, cancel) — not a synchronous request.
5. **No way to *find* L6 evidence afterward.** See §4: the L6 card, the `L6` filter chip, and even the `gazebo_l6` track chip don't surface the Gazebo runs (mislabelled `student`).

**Fix C (design):** add an explicit campaign selector to the Rollout form, backed by a new endpoint (proposed, not applied):

```html
<!-- index.html — Rollout Trigger -->
<label>Bridge <select name="bridge"><option value="fake">fake (fast sim)</option><option value="gazebo">gazebo (L6, Harmonic)</option></select></label>
<label>Sensing <select name="sensing"><option>emulated</option><option>camera</option><option>camera_dr</option></select></label>
```

```python
# server.py (proposal) — gazebo branch of /api/rollout
if bridge == "gazebo":
    npz = RL_RUNS / (Path(policy_id).stem + ".npz")   # or run rl/export.py on demand
    cmd = [str(py), str(REPO / "simulation/gazebo/run_policy_episodes.py"),
           "--bridge", "gazebo", "--weights", str(npz), "--episodes", str(episodes),
           "--level", level, "--seed", str(seed), "--sensing", sensing,
           "--save-traces", str(trace_path),
           "--json", str(RL_RUNS / "gazebo_l6" / f"obs_{ts}.json")]
    cwd = str(REPO)                                    # runner resolves REPO itself
```

and post-run, the result must land in `rl/runs/gazebo_l6/` so `_counts()`/`_leaderboard()`/`replay_list()` see it as `source=gazebo_l6`.

### 2.4 Also visible in the same screenshot

- The status line `job job_1794673578 : failed` is the **only** feedback; there is no spinner/disabled state on the Run button while the job is in flight (double-click ⇒ duplicate 5-episode rollouts, `obs_<same ts>` collisions are possible within the same second).
- `level` was `L1` with `student_L2_cons_adr_s0.pt` — a sensible request the user had every right to make; nothing was wrong with their inputs.

---

## 3. Bug #2 — Replay tab: "No pose in this frame" / "No frames" on `student_L2_cons_adr_s0_L2_cons`

The bundle itself is **fine** (verified: `rl/runs/traces/student_L2_cons_adr_s0_L2_cons.json`, `format=ugv_replay_v1`, 20 episodes, 84–2250 frames each, `true_pose` triples; `eval_bundle()` normalizes it cleanly — `ep0 final_error=1517.3 mm`, `tag_age=49999998976` sentinel). The emptiness is produced by the **frontend state machine**, in four compounding ways.

### 3.1 Failed-rollout auto-load poisons the tab (see §2.2)

`app.js:1104` loads `obs_1794673578` on failure → 404 → `state.episodes=[]` → `drawReplay()` prints "No pose in this frame" (`app.js:783`) and `drawStrips()` prints "No frames" (`app.js:954-957`) → empty `#episode-list`. The canvas, transport, strips and list all go blank **without a single error message**. This is the exact state in screenshot `5ab81dda….png` (note `step 0` with no `/N`, vs. `89dceecf….png` which shows `step 0/479` with the same bundle properly selected — confirming the same tap-list works until the failed job wipes it).

**Fix D (fail loudly, keep last-good state):**

```js
async function loadBundle(id){
  const prev = {bundle: state.bundle, episodes: state.episodes, currentEp: state.currentEp, frameIdx: state.frameIdx};
  const d = await fetchJSON('/api/replay/bundle/'+encodeURIComponent(id));
  if(!d){
    renderEpError(`Bundle “${id}” could not be loaded (404/500 — see server log or /api/replay/bundle/${id}). Keeping previous bundle.`);
    Object.assign(state, prev);          // do NOT wipe a good bundle on a bad fetch
    return;
  }
  ...
}
```

`fetchJSON` should take an `onError` or the call sites should check `res.ok` and surface the status — today it maps *every* failure (404, 500, network, invalid JSON) to `null` (`app.js:36`).

### 3.2 Bundle schema mismatch is not detected: "valid-looking" bundles with zero frames

If the user selects a Gazebo raw summary (e.g. `gazebo_l6:baseline_gazebo_L1`, which `/api/replay/bundle` will happily normalize because it falls through to `eval_bundle()`), the JSON parses fine but **has no `frames`**: `baseline_gazebo_L1.json` carries `episodes: [{start, result, t, pose, phase}, …]` — no `frames` key. `_norm_episodes()` (`replay.py:317-329`) iterates them, sets `ep["frames"]=[]`, `duration=0.0`, `final_error=None`. Result: 100 episodes in the list, **every frame canvas empty**, strips say "No frames" — indistinguishable from Bug #3.1 to the user. Similarly `obs_*.json` files (e.g. `obs_1794669415.json`, visible in the bundle dropdown in screenshot `9c780e5a….png`) are evaluate.py **metrics** files with `episodes: <int>` — `_norm_episodes` leaves `episodes` as an `int`, `state.episodes = d?.episodes || []` → `episodes` becomes `20` (a number!), `.length` undefined … `renderEpList(20)`'s `forEach` throws, tab half-renders.

**Fix E (schema gate in the UI, and a source-aware API):**

```js
const eps = Array.isArray(d?.episodes) ? d.episodes
          : (Array.isArray(d?.poses) ? [synthEp0(d)] : []);
if(d && !eps.length){
  renderEpError(`Bundle “${id}” contains no replay frames (source=${d.source||'?'} — likely a metrics/summary file, not a trace). Pick a trace bundle (…traces.json / l7:*).`);
}
```

and server-side, `replay_list()` should stop advertising non-trace artifacts: `RL_RUNS.glob("*.traces.json")` and `traces/*.json` only — never emit a bundle entry for `obs_*.json`/`*_1000ep_s11.json` summary files, or tag them `kind:"summary"` and have the UI hide/label them.

### 3.3 `epFilter` loses state on every bundle switch and has a broken matcher

```js
function renderEpisodeList(){ ... filter buttons rebuilt, 'All' re-activated ... }   // app.js:367-372
function filterEpisodes(f){
  eps = state.episodes.filter(e => (e.outcome||'').toLowerCase()
        .includes(f.toLowerCase().replace('-park','park')));                        // app.js:373-377
  renderEpList(eps);
}
```

Problems:
1. **Filter is not in `state`** (there is no `state.epFilter`). Every `loadBundle()` rebuilds the five buttons and hard-selects `All`, discarding the user's `Collision` filter — the exact interaction in the bug report ("when the user filtered or switched bundles"). Selection of the *list* changes silently; the canvas keeps whatever `state.currentEp` is (index into the unfiltered array — kept correct only by accident of `indexOf`).
2. **`False-Park` can never match.** `'False-Park'.toLowerCase().replace('-park','park')` → `'falsepark'`; the recorded outcome is `'false_park'` (underscore) — `'false_park'.includes('falsepark') === false`. The button is dead UI; it always renders an empty list, which the user experiences as yet another "no data" state.
3. **Matching by substring over outcomes** is fragile across sources: sim traces use `success|collision|timeout`, L7 summaries use `parked|success|…` (`server.py:206`), Gazebo raw prints `result: success|timeout|collision`. Normalize outcome to a canonical enum (`success|collision|timeout|false_park|unknown`) at the `_norm_*` boundary and compare exactly.
4. **Filter doesn't re-clamp the selection.** If the filtered list is empty (or the current episode is no longer in it), `#episode-list` shows nothing while the canvas/transport still display a stale episode — no "no episodes match this filter — showing all" fallback.

**Fix F:**

```js
state.epFilter = 'All';
function renderEpisodeList(){
  ['All','Success','Collision','Timeout','False-Park'].forEach(f => button(f, () => {
    state.epFilter = f; paintActive(); renderEpList(applyEpFilter(state.episodes));
  }, f===state.epFilter));
}
const OUTCOME = { 'false_park':'falsepark', 'false park':'falsepark', 'parked':'success', /* … */ };
const canon = o => (OUTCOME[String(o||'').toLowerCase().trim()] ?? String(o||'').toLowerCase().trim().replace(/[-_\s]/g,''));
const applyEpFilter = eps => state.epFilter==='All' ? eps
  : eps.filter(e => canon(e.outcome).includes(canon(state.epFilter)));
```

and when the current episode is filtered out, select the first visible episode (or show an explicit empty-state row) instead of leaving the transport pointing at an invisible item.

### 3.4 `steps_since_valid` sentinel renders as a 50-billion-second tag age

`_norm_frame()` passes `steps_since_valid` straight to `tag_age` when `tag_age` is absent (`replay.py:285-289`). This bundle emits `49999998976` (never-valid sentinel). `drawStrips()` (`app.js:963-977`) takes `max(|tag_age|)` over the episode as the normalizer — one sentinel frame ⇒ every other frame's tag-age line flattens to ~0 and the numeric readout in the camera POV shows `age=49999998976s` (`app.js:840`). Treat anything ≥ 2³¹ as `∞`/`None` at the normalize boundary (`tag_age = None` when `steps_since_valid >= 2**31`) and render `—`.

### 3.5 What the user should have seen, end to end

Trigger a rollout → `202` + job id immediately → status poll shows `running… (elapsed, episodes done)` → on `done`, tab switches to the fresh bundle with a toast naming it; on `failed`, the L2_cons bundle stays on screen with a red, actionable message. Nothing in that path may call `loadBundle()` on a nonexistent id, and no canvas may ever say "No pose" while a valid bundle is selected without saying *why*.

---

## 4. Bug #3 — L6 and L7 frontier cards permanently read `—`

### 4.1 Root cause: the cards filter on a field nothing has

`app.js:108-113`:

```js
['L0','L1','L2','L3','L_real','L_chaos','L6','L7'].forEach(l=>{
  const subset = state.leaderboard.filter(r => r.level === l);   // ← hard equality on `level`
```

Runtime truth (`_leaderboard()` executed against this repo):

| Group | What the entries actually carry | Card predicate finds |
|---|---|---|
| Gazebo L6 runs (`rl/runs/gazebo_l6/*.json`, 12 parsed entries) | `level: "L1"`/`"L2"`/`None`, `source: "gazebo_l6"`, `track: "gazebo_l6"` **only if** the run_id lacks "student" (6 of 12) | **zero** matches for `L6` → card `—` |
| e.g. `student_L2_cons_adr_s0_gazebo_L1_5ep` | `level=L1, track=student, source=gazebo_l6, success=0/5` | counted as "student", never L6 |
| L7 field runs (25) | `server.py:210-212` **hardcodes** `track: "l7"`, `level: "L_real"`, `episodes: 1` | **zero** matches for `L7` → card `—` |

Meanwhile the chip row (`index.html` has no explicit markup; `buildFilters()` `app.js:27-32`) renders an `L6` chip from the frontend `LEVELS` const (`app.js:23` — which contains `L6` but not `L7`), so clicking it filters `level==='L6'` → empty table, and there is no `L7` chip at all (screenshots `6d553340`, `f0d16211`). The backend confirms: `data.py:12 LEVELS = ["L0".."L5","L_real","L_chaos"]`; no `L6`/`L7` anywhere server-side.

### 4.2 Second defect hiding behind it: `_track_for` mislabels Gazebo runs

`server.py:86-99`:

```python
def _track_for(d, source, run_id):
    track = d.get("track")
    if track: return track
    rid = (run_id or "").lower()
    if "student" in rid: return "student"     # ← fires before the gazebo check
    ...
    if source == "gazebo_l6": return "gazebo_l6"
```

**Every student Gazebo run is labelled `track=student`.** Verified at runtime: `track` counter = `gazebo_l6:6` (all non-student), while 6 student-gazebo runs (`student_L2_cons_adr_s0_gazebo_L1_3ep/5ep`, `student_L1_latw5_s0_*` ×3, `student_L2_cons_adr_s0_camera_dr_L1_20ep`) count as `student`. So even the existing `gazebo_l6` track chip shows only 6 rows and misses exactly the runs the user cares about (`baseline_gazebo_L1` = 97% is visible; `student_L2_cons_adr_s0_gazebo_L1_5ep` is not). The correct precedence is **source > explicit track > name-sniffing**:

```python
def _track_for(d, source, run_id):
    t = d.get("track")
    if t: return t
    if source == "gazebo_l6": return "gazebo_l6"      # hardware campaign outranks archetype
    if source == "l7":        return "l7"
    rid = (run_id or "").lower()
    if "student" in rid: return "student"
    if "teacher" in rid: return "teacher"
    return "baseline"
```

### 4.3 Semantics: "frontier card = max(success_rate) among N runs" is the wrong estimator for L7

Even after the predicate is fixed, `best = max by success_rate` (`app.js:110`) over L7's n=1-per-run entries will headline "100% succ" for whichever single L7 run succeeded — a Wilson CI [0.21, 1.00] presented as a frontier result. For the L6/L7 cards use an **aggregate** over the campaign — `Σsuccess / Σepisodes` with a Wilson interval over the pooled counts (e.g. L6: 97/100 on `baseline_gazebo_L1` + 0/25 across the student L1/L2 gazebo runs = 97/125 pooled), and show the count (`13 runs · pooled 78%`), not a lone lucky run. The table's per-run rows stay as-is.

**Fix G (frontend, exact):**

```js
// app.js — replace the cards block (lines ~107-114)
const cardDefs = [
  {label:'L0', f:r=>r.level==='L0'}, … {label:'L_chaos', f:r=>r.level==='L_chaos'},
  {label:'L6', f:r=>r.source==='gazebo_l6' || r.track==='gazebo_l6'},
  {label:'L7', f:r=>r.source==='l7'        || r.track==='l7'},
];
cardDefs.forEach(({label, f}) => {
  const subset = state.leaderboard.filter(f);
  const succ = subset.reduce((a,r)=>a+(r.success??0), 0);
  const eps  = subset.reduce((a,r)=>a+(r.episodes??0), 0);
  const agg  = eps ? succ/eps : null;
  // primary line: `${agg!=null ? (agg*100).toFixed(0)+'% succ' : '—'}  (Σ ${succ}/${eps})`
  // sub-line:    `${subset.length} runs · best ${best?.run_id ?? '—'}`
});
```

and add `'L7'` to the frontend `LEVELS` const (or, better, drive the chip row from `/api/overview` + a real Campaign dimension: `ALL | student|teacher|baseline | gazebo_l6 | l7 | archive | eval_json`), dropping the meaningless `L6` **level** chip in favour of a `gazebo_l6` **campaign** chip that now matches all 12 runs after Fix G1/G2.

### 4.4 Data-hygiene bugs in the same code path (fix while here)

1. `_counts()` gazebo_l6 counter (`server.py:50-53`) counts the `baseline_L1_20ep_s11.json` **directory** and `parallel_backlog_plan.json` (a plan list, `bridge`-less, `[...]`): runtime shows **14** where 13 files exist and only ~11 are actual Gazebo bridge runs. Require `p.is_file()`, exclude top-level lists (`isinstance(json.load(p), dict)`), and preferably test `d.get("bridge")=="gazebo"` / `d.get("engine","").startswith("gazebo")`. The 29-byte `student_L2_cons_adr_s0_camera_dr_L1_20ep.json` has `bridge:"fake"` and also counts as a "Gazebo L6 run" today.
2. `evals = len(RL_RUNS.glob("*.json"))` (`server.py:43`) likewise counts `pi_inference_bench.json` and `obs_*.json` rollout outputs as "Eval runs"; reported 77 vs. the `archive 44 + gazebo 13 + ~20 real evals` the user expects. At minimum exclude `pi_inference_bench.json`; label the pill `Eval artifacts` or filter to files with `success`+`episodes`.
3. In the gazebo merge (`server.py:161-203`), a raw flat run JSON (`student_L2_cons_adr_s0_gazebo_L1_5ep.json`) takes `time_median=4.56` etc. — fine — but its `false_parked` key is absent so `_pick("false_parked")` → `None` → `_norm_entry` defaults `collisions` etc. to 0 silently: **absent ≠ 0**. Rows with `success=None` are correctly rate-less (`server.py:291-294`), but absent `collisions` should render `—`, not `0`, or the table "lies green" for partial records.
4. `baseline_gazebo_L1.json` has `summary.level="L1"`, no top-level `bridge`, engine string `gazebo-harmonic native mecanum rollers`, and `episodes` as a **list of 100 dicts** nested under `summary.episodes`. The merge at `server.py:163-173` keeps scalar `summary.episodes=100` over the per-episode list, and `success=97`, so the L1 card reads `97% succ` — but this silently depends on the `_pick`/merge ordering. Add a unit test pinning `baseline_gazebo_L1.json → (episodes=100, success=97, track=gazebo_l6 after §4.2 fix)`.

### 4.5 What the L6/L7 cards should show (acceptance criteria)

- **L6** card: `97% pooled` over `baseline_gazebo_L1 (97/100) + student gazebo runs (0/30 incl. 3ep/5ep/20ep)` → `97/130`, Wilson CI shown, sub-line `12 runs · seeds 11 · levels L1/L2`.
- **L7** card: `x/25 parked` pooled from `calibration/runs/l7/*/summary.json (outcome)`, Wilson CI, sub-line `25 runs · latest 20261113-181848`.
- Clicking a card applies the corresponding campaign filter to the table (cards currently are not clickable — missed affordance).

---

## 5. Other workflow failures visible in the screenshots

### 5.1 Training & ADR Curves — legends overdraw each other (`920f3e4b….png`, `51e85898….png`)

`renderChart()` legend block (`app.js:267-273`):

```js
let lx = padL + 4;
lines.forEach(l => {
  ctx.fillRect(lx,4,8,8); … ctx.fillText(l.name, lx+11, 12); lx += w + 22;
  if (lx > W - 80) { lx = padL + 4; }   // ← wraps X but NOT Y — everything redraws at y=4…12
});
```

With ≥6 lines per metric set × several runs checked, all wrapped entries repaint on the same top row — the illegible rainbow smear at the top of every chart in the screenshots. Fix: maintain a `ly` row index (`if(lx > W-160){lx = padL+4; ly += 14;}` and increase `padT` when wrapping), or render legend as an HTML overlay list with ellipsis + `title` tooltips. Also:

- **Chart 4 title/spec mismatch:** title says "std & entropy / mean_return & sps" (`index.html:63`), `CHART_SPECS.chart4` plots only `std` and `sps` (`app.js:210`). Either add `entropy, mean_return` to the spec and plot all four on auto-scaled axes (draw secondary axis or normalize), or fix the title.
- 3 curves per metric × 5 metrics = 15 simultaneous lines in chart 1 is unreadable regardless of legend placement; default-checked runs should be ≤2, and the selector (`loadCurvesRuns`, `app.js:182-190`) already caps checks at 3 but mixes `(legacy .log)` and `(metrics.jsonl)` formats with no grouping — sort groups headers by format and label the archive dir.
- `n_iters` for logs counts lines containing "iter" anywhere (`server.py:347`) — inflates on wrapped log lines; cosmetic but misleading on the badge.

### 5.2 Leaderboard filter row UX

- Two chip rows with duplicate `ALL` (`LEVELS` and `TRACKS` both lead with ALL, `app.js:23-24`) — ambiguous which ALL is active; use segmented control semantics (†one active per group) — currently both groups highlight independently, fine, but the groups are unlabeled (`LEVELS` vs `CAMPAIGN`). Add group labels.
- `L6` chip exists in the frontend `LEVELS` but **no `L7` chip**; after §4 lands, chips should be generated from server-provided facets (`level` + `campaign`), not a hand-maintained const duplicated in two files.
- Search box is `run_id`-only (`app.js:82`); matching `track`/`level`/`tags` costs nothing and matches user intent ("student_L2_cons_adr*").

### 5.3 Sim-vs-Real Contract — INFO is rendered as a red failure (`9810da75….png`)

`loadContract()` (`app.js:1064-1066`):

```js
banner.className = 'banner ' + (String(d.status).toLowerCase().includes('pass') ? 'pass' : 'fail');
```

`_eval_contract()` sets `status = verdict or ('PASS' if passed else 'INFO')` (`server.py:725`) → an **INFO** report (metrics missing, sim reference unavailable, or non-gate watch run `20261113-181848-watch-shutter4ms` with `valid_pose_rate=0`) gets the same red styling as a hard FAIL. The screenshot's red INFO band reads as "the contract failed", which the user cannot distinguish from a real gate breach. Add `.banner.info{background:#2a2a1a;color:var(--warn)}` and a third branch; likewise render `check.pass===null` (unknown) as grey, not red — `_norm_contract_checks` synthesizes only for `real_metrics` fallback and returns pass booleans derived from thresholds the user never calibrated for watch-mode runs (`valid_pose_rate>=0.5` labelled `>=0.35` in the card subline — inconsistent limit strings between `_norm_contract_checks` `>= 0.50` and the card's `>= 0.35` shown in the screenshot — pick one source of truth). Also the channels table renders `sim μ=50.0 σ=0.0` nominal fallbacks without flagging them as fabricated — set `sim_imputed:true` and render "(nominal)".

### 5.4 Job telemetry / lifecycle

- `JOBS` grows forever (`server.py:34`) and `rollout_status` leaks absolute server paths (`json_path`, `server.py:541`). Cap the map (LRU, e.g. 50), TTL 1 h, drop `json_path` from the public response.
- No cancel endpoint — a Gazebo L6 job (§2.3) is ~10 min/episode × up to 20 episodes; without `/api/rollout/{id}/cancel` the user cannot stop a runaway rollout without killing the server.
- `auto-refresh` (`app.js:1134-1136`) re-fetches leaderboard every 5 s and **rebuilds the filter buttons + cards**, nuking scroll/focus and any in-progress Diff selection interactivity on each tick — re-render in place or only when data actually changed (compare a cheap `ETag`/hash).
- `renderLeaderboard()` rebuilds the whole table on every keystroke in search (`app.js:34,76`) — fine at 100 rows, but pair with `tbody` document fragment (already does `innerHTML=''` then appends — acceptable) and debounce 150 ms.
- `L7` table (`renderL7Table`) fetches `/api/l7/<id>/end.jpg` lazily per row only via Replay; add `loading="lazy"` thumbnails in the row itself — the field run's end frame is the single most diagnostic artifact for L_real and it is one click away but never visible.

### 5.5 Replay tab polish (same area as §3)

- `loadRolloutOptions()` overwrites the Policy select's `<option value="student">` placeholder silently; if the fetch fails the form keeps a bogus `student` entry that is not a loadable checkpoint — same class of bug as §2.1. Render an explicit empty/error option instead.
- When the episode list is empty, the Transport buttons stay enabled and `Play` toggles `state.playing` with no effect (`app.js:1025`) — disable them.
- `#scrubber` uses `max=frames.length-1`; after Bug #2.1 the max stays `0` but the slider is still draggable-styled; set `disabled`.
- `drawReplay()` reads `geo.tags` fallback `[{id:14…}]` — the **older 9/12/13/14 constellation** comment in AGENTS.md is mirrored here with 14/6/3 (current) — fine — but `drawStrl001Airframe` draws hardcoded tags `#20/#21/#22` (`app.js:591-596`) which are **not in any documented tag layout**; if STRL001 ships different ids, this silently mislabels. Move STRL001 tag ids into the bundle's `geometry.tags` and render from data, else the canvas "verifies" a constellation that does not exist on the airframe.

### 5.6 Unrelated captures (no action)

`2feb11ba….png` (RealFlight 9.5 error dialog) and `6ae36441….png` (huggingface Cosmos3 page) are not Observatory artifacts; citing them as Observatory failures would be fabrication. Note them in the board as "out of scope" so the next critic does not chase them.

---

## 6. Fix plan (priority order)

| # | Fix | Where | Risk | Verified by |
|---|---|---|---|---|
| P0 | §2.1 Fix A — resolve `.pt` to absolute `rl/runs/` path; reject unknown policy ids with 400 | `server.py` `/api/rollout` | none (strictly stronger validation) | `curl -X POST /api/rollout -d '{"policy":"student_L2_cons_adr_s0.pt",…}'` → job `done`, trace written |
| P0 | §2.2 Fix B — capture child stdout/stderr to `*.job.log`; return `log_tail` on failure; don't `loadBundle()` on failed jobs | `server.py`, `app.js` rollout submit | low | trigger rollout against missing `.npz` → UI shows torch `FileNotFoundError` line |
| P0 | §3.1 Fix D / §3.2 Fix E — error banner on bundle load failure; schema gate; `replay_list` only advertises trace bundles; never emit `obs_*.json` metrics as bundles | `app.js`, `server.py:replay_list` | low | select `obs_1794669415` → banner, previous bundle retained |
| P1 | §4 Fix G (+G1 `_track_for` reorder, +§4.4 counting fixes) — cards filter `source/track` not `level`; pooled aggregate display; `L7` chip added; chips driven by server facets | `app.js`, `server.py` | low | L6 card shows 97/130 pooled; L7 card shows x/25; `gazebo_l6` chip shows 11 rows |
| P1 | §3.3 Fix F — `state.epFilter`, canonical outcome matching, False-Park works, selection re-clamped | `app.js` | low | click False-Park on `student_L2_cons_adr_s0_L2_cons` → 0 rows *shown intentionally* with empty-state note |
| P1 | §5.3 INFO banner class; check unknown→grey | `app.js`, `style.css` | cosmetic | contract tab on watch-shutter run → amber banner |
| P2 | §2.3 Fix C — Gazebo bridge/sensing campaign in Rollout form + `/api/rollout` gazebo branch (`.pt→.npz` export guard, 10-min jobs, cancel endpoint) | `server.py`, `index.html`, `app.js` | **medium-high** (GPU/Display; off by default) | `smoke` 1-episode gazebo job writes `rl/runs/gazebo_l6/obs_*.json`, appears under `gazebo_l6` chip |
| P2 | §5.1 legend row-wrap fix; chart4 spec/title reconcile; ≤2 default runs | `app.js`, `index.html` | cosmetic | charts legible at 100% zoom |
| P2 | §3.4 `steps_since_valid ≥ 2³¹ → null`; `tag_age=∞` renders `—` | `replay.py`, `app.js` | low | strips tag_age line non-degenerate on L2_cons |
| P3 | §5.4/§5.5 job map LRU+TTL, drop `json_path` from API, auto-refresh diffing, disable transport on empty episode list, render STRL001 tags from `geometry.tags` | `server.py`, `app.js` | low | — |

**Non-goals / do not touch:** `docs/SAFETY.md` levels, `config/parking.json`, gains/caps/watchdog, the `pose2` frame boundary, `.npz` on the Pi, `HANDOFF.md` state. A real Gazebo L6 trigger (P2) still requires a human-approved clear zone per `docs/SAFETY.md` before anyone lets a policy move the **real** car — L6 is the Gazebo proving ground, not a license to bypass the field gate.

## 7. Summary for the board

1. **Rollout Trigger never worked.** Bare filename + `cwd=rl/` ⇒ `torch.load` FileNotFoundError on every checkpoint; stderr discarded; UI prints only `job … : failed`; the frontend then auto-loaded the nonexistent trace and blanked the Replay tab (this is the "No pose / No frames" the user saw on `student_L2_cons_adr_s0_L2_cons`).
2. **Gazebo L6 is invisible-in, invisible-out.** No bridge control in the dashboard, no Level entry for it (correctly — it's a campaign, not a sim level), `.pt` checkpoints offered where `.npz` exports are required, and once runs exist they are mislabelled `student` (6/12), never `L6`, excluded from the L6 card and `L6` chip, and overcounted in the header pill (14 incl. a directory and a plan file).
3. **L7 field evidence is present but unelectable.** 25 runs are loaded (`track=l7`, `level=L_real`, n=1 each) yet the L7 card hard-filters `level==="L7"` and there is no `L7` chip — a one-word predicate bug, twice.
4. Every one of these is reproducible from the code as written; §6 P0/P1 fixes are each under ~30 lines and need no new dependencies.
