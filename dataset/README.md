# Shared-control pHRI trials — FR3 circle-tracing, N=3

Trial-level data (per-control-cycle CSV logs + optional rosbags) from the
`sc_ros_empathic` reactive shared-control experiment: a volunteer
physically guides a Franka FR3 end-effector around a circular path while
a controller blends the human's intent with an autonomous path follower,
weighted by up to four performance factors (smoothness, directness,
human joint-limit safety, robot manipulability). Three volunteers
(S01–S03), each run through every applicable experimental condition
(A–F) in both a nominal and a per-participant "stressed" placement.

Software: <https://github.com/TaISLab/sc_ros_empathic> (or the lab's
private remote — fill in the actual URL before publishing), commit
`802d752` at export time. **Data collected before commit `1d0c4f9`
(2026-09-09) used an earlier, buggy version of the human joint-angle
convention and the `joint_safety` factor formula — do not pool it with
later trials.** See `docs/paper_update_2026-09-07_to_09.md` and
`docs/joint_safety_factor.md` in the software repository for the exact
formula and what changed.

## Subjects

| id | date | sex | age | height (m) | mass (kg) | l1 (m) | l2 (m) | notes |
|---|---|---|---|---|---|---|---|---|
| S01 | | | | | | | | |
| S02 | | | | | | | | |
| S03 | | | | | | | | |

Fill in from each `config/subjects/SXX.yaml` on the lab machine (only
the anonymised id is used here — no names). `l1`/`l2` are the upper-arm
/ forearm segment lengths used to seed the arm model (they also vary
slightly per trial; see the `l1,l2` CSV columns and the sidecar).
Per-subject pre-registered joint ranges (`joint_limits_rad`), if
calibrated with `joint_range_probe.py`, are in each `SXX.yaml` — copy
them into a `subjects/SXX.yaml` folder alongside the trials if you want
them archived here too (optional; they are already embedded in every
trial's `.params.json` sidecar).

## Experimental conditions

Every candidate command (human `v_h`, robot `v_r`, blend) gets an
efficiency `eta` in `[0,1]`, a weighted mean of the *active* factors;
the condition sets which factors are active and whether the robot's
path-following command participates at all.

| id | active factors | role | placement(s) collected |
|---|---|---|---|
| `A_standalone` | none | own-drive baseline: only `v_h` shaped, no assistance | nominal + stressed |
| `B_baseline_m2` | smoothness, directness | reactive shared control of Ruiz-Ruiz et al. [1] as-is — the **reference** condition | nominal + stressed |
| `C_jointsafety_m3` | smoothness, directness, joint_safety | ablation: baseline + only the joint-limit-safety factor | stressed (optional) |
| `D_manip_m3` | smoothness, directness, manipulability | ablation: baseline + only the manipulability factor | stressed (optional) |
| `E_extended_m4` | all four | **the proposed controller**, compared against B | nominal + stressed |
| `F_impedance_aan` | *(separate controller)* | impedance-control assist-as-needed baseline of Zhang et al. [9] | nominal only |

Each trial is 4 laps (`trial_laps:=4`), the first discarded as
training (`lap == 0` in the CSV) — 3 analysed laps per trial.

**Placement.** *nominal* = safe mid-workspace circle, homogeneous
across volunteers (`path_center=[0.45,0,0.45]`, `path_radius=0.05`).
*stressed* = near the edge of the volunteer's own combined human+robot
reach (~85–90 %), calibrated per participant — the `path_center` in
the sidecar therefore differs subject to subject even though the label
is the same.

## Files

One CSV + one JSON sidecar per trial, named
`<subject>_<condition>_<placement>_<YYYY-MM-DD_HH-MM-SS>.csv` /
`....csv.params.json`, grouped under `S01/`, `S02/`, `S03/`. Optional
rosbags (full topic recording, see below) are large and kept separately
under `bags/` as one `.bag.zip`/session (not one per trial — recording
spans the whole session, sliced offline by lap).

- **`<trial>.csv`** — one row per control cycle (~100–200 Hz):
  `t` (epoch s), `t_rel` (s from first row), `wall_time`, `condition`,
  `lap`, `s_near`, `cross_track`, `px..pz` (EE position), `vh_*`,
  `vr_*`, `vs_*` (the three Cartesian velocity terms), `fx..fz`
  (interaction force), `human_fresh` / `jac_fresh` (0/1 validity
  flags), `eta_h, eta_r, eta_s`, `smoothness_h, directness_h,
  joint_safety_h, manip_h` (the four partial factors for the human
  candidate), `rho1..rho4` (signed human joint position in `[-1,1]`),
  `m1..m4, m_min` (joint margins), `w_qr` (robot manipulability index),
  `l1, l2` (arm segment lengths used that cycle, m). `joint_safety_h`
  and `m1..m4` are `NaN` on cycles where `human_fresh==0`. Condition F
  logs a reduced subset (no factors/manipulability/joint terms — that
  controller does not compute them).

- **`<trial>.csv.params.json`** — full run configuration: path
  geometry, follower gains, factor weights + `Cs` +
  `proximity_threshold`, `planar_task`, the resolved human joint
  limits (rad) actually used that trial + `human_joint_offsets` /
  `human_joint_gains`, static `l1,l2` fallback, admittance parameters,
  topic names, and the exact git commit of the software that produced
  the trial. **This is what makes a CSV self-contained** — without it,
  `rho_i` cannot be traced back to physical joint angles. Always keep
  the two files together.

- **`bags/<session>.bag.zip`** *(optional, if recorded)* — full topic
  recording for a session (one bag can hold several trials/laps back
  to back; slice by the `path_progress` lap index). Topics: FR3 state,
  human joint states, arm points, the emergent velocity command, and
  every `~diag/*` / `~eta` observable — see the software README for the
  full per-topic table.

## Reading the data

Any CSV loads directly with `pandas.read_csv`, no dependencies beyond
the JSON sidecar for context:

```python
import pandas as pd, json
df = pd.read_csv("S01/S01_E_extended_m4_stressed_2026-09-08_11-04-12.csv")
params = json.load(open("S01/S01_E_extended_m4_stressed_2026-09-08_11-04-12.csv.params.json"))
laps = df[df["lap"] >= 1]          # drop the training lap
```

The software repository's `matlab/plot_joint_angles_offline.m` and
`matlab/plot_paths_offline.m` reproduce the paper's joint-angle and
traced-path figures directly from a CSV + its sidecar, if MATLAB is
available; they are not required to use the data.

## Known limitations / caveats

- Ablation conditions C and D were collected opportunistically and may
  be missing for some subjects (see `MANIFEST.csv`).
- Gestures/forces are not matched across trials or subjects (natural
  tracing, not a scripted force profile) — expect `|f|` to vary trial
  to trial.
- The manipulability factor (`manip_h`, `w_qr`) penalises
  *degradation* of the robot's manipulability index, not low absolute
  values — it does not, by design, prevent the FR3 approaching its own
  reach/singularity boundary in the stressed placement.
- Any trial timestamped before 2026-09-09 predates the q3/q4
  joint-angle convention fixes and the gated/signed `joint_safety`
  formula (see the commit note above) — treat separately from later
  data if mixed in this deposit.

## License

Data: CC-BY 4.0 (proposed — confirm before publishing). Software:
MIT, Telerobotic and Interactive Systems Laboratory (TaISLab),
University of Málaga — see the software repository's `LICENSE`.

## Citation

*(add the paper citation once accepted/published)*

## Contact

J. Manuel Gómez-de-Gabriel — TaISLab, University of Málaga —
jesus.gomez@uma.es
