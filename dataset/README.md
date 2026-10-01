# Shared-control pHRI trials — FR3 circle-tracing, N=3

Trial-level data (per-control-cycle CSV logs + optional rosbags) from the
`sc_ros_empathic` reactive shared-control experiment: a volunteer
physically guides a Franka FR3 end-effector around a circular path while
a controller blends the human's intent with an autonomous path follower,
weighted by up to four performance factors (smoothness, directness,
human joint-limit safety, robot manipulability). Three volunteers
(S01–S03), each run through conditions A–E in both a nominal and a
"stressed" placement (condition F collected but not yet included —
see Curation).

Software: <https://github.com/TaISLab/sc_ros_empathic> (or the lab's
private remote — fill in the actual URL before publishing), commit
`802d752` at export time. **Data collected before commit `1d0c4f9`
(2026-09-09) used an earlier, buggy version of the human joint-angle
convention and the `joint_safety` factor formula — do not pool it with
later trials.** See `docs/paper_update_2026-09-07_to_09.md` and
`docs/joint_safety_factor.md` in the software repository for the exact
formula and what changed.

## Subjects

| id | date | sex | age | height (m) | mass (kg) | l1 (m) | l2 (m) | source |
|---|---|---|---|---|---|---|---|---|
| S01 | *n/r* | m | 60 | 1.82 | 85 | 0.29 | 0.27 | tape |
| S02 | *n/r* | m | 46 | 1.75 | 108 | 0.33 | 0.29 | tape |
| S03 | *n/r* | f | 54 | 1.75 | 75 | 0.34 | 0.30 | tape |

Only the anonymised id is used here — no names. `date` was left blank
in the source subject files (*n/r* = not recorded); fill in per-session
dates if/when reconstructed from the trial filename timestamps. `l1`/
`l2` (upper-arm / forearm segment length) are the static values used to
seed the arm model; the live per-cycle value used by the controller
that trial is in the `l1,l2` CSV columns and may drift slightly from
this table (visuo-tactile re-estimate vs. tape measurement here).

Pre-registered joint ranges (`joint_limits_rad`, rad), from
`joint_range_probe.py` ROM sweeps, all four required if the block is
present at all:

| id | q1 (shoulder flex/ext) | q2 (shoulder abd/add) | q3 (shoulder int/ext rot) | q4 (elbow flex/ext) |
|---|---|---|---|---|
| S01 | [-0.566, 1.563] | [-0.081, 1.408] | [-1.57, 1.57] (study default) | [0.001, 2.770] |
| S02 | [-0.310, 1.829] | [-0.204, 1.389] | [-1.577, 1.800] | [0.199, 2.798] |
| S03 | [-1.018, 1.496] | [-0.610, 1.471] | [-1.57, 1.57] (study default) | [0.156, 2.512] |

S01's and S03's q3 ranges are the study default (commented out /
unset in the source subject file), not a per-participant calibration —
the same `[-1.57, 1.57]` used when the block is omitted entirely; only
S02 has a directly calibrated q3. These are the ranges `rho1..rho4`
are normalised against in every trial where `subject:=SXX` was passed;
they are also embedded per-trial in each `.params.json` sidecar, which
is the authoritative source if this table and a given trial ever
disagree (e.g. after a later re-calibration).

## Experimental conditions

Every candidate command (human `v_h`, robot `v_r`, blend) gets an
efficiency `eta` in `[0,1]`, a weighted mean of the *active* factors;
the condition sets which factors are active and whether the robot's
path-following command participates at all.

| id | active factors | role | placement(s) collected |
|---|---|---|---|
| `A_standalone` | none | own-drive baseline: only `v_h` shaped, no assistance | nominal + stressed |
| `B_baseline_m2` | smoothness, directness | reactive shared control of Ruiz-Ruiz et al. [1] as-is — the **reference** condition | nominal + stressed |
| `C_jointsafety_m3` | smoothness, directness, joint_safety | ablation: baseline + only the joint-limit-safety factor | nominal + stressed |
| `D_manip_m3` | smoothness, directness, manipulability | ablation: baseline + only the manipulability factor | nominal + stressed |
| `E_extended_m4` | all four | **the proposed controller**, compared against B | nominal + stressed |
| `F_impedance_aan` | *(separate controller)* | impedance-control assist-as-needed baseline of Zhang et al. [9] | **not in this deposit — see Curation** |

Each trial is `trial_laps:=7`, the first discarded as training
(`lap == 0` in the CSV) — 6 analysed laps per trial.

**Placement.** *nominal* = safe mid-workspace circle, homogeneous
across volunteers (`path_center=[0.45,0,0.45]`, `path_radius=0.05`).
*stressed* = near the edge of the volunteer's own combined human+robot
reach, `path_radius=0.10`, offset towards the participant
(`path_center` in the sidecar). The exact offset is **not held fixed
across sessions for the same participant** — seat/chair height was not
standardised during data collection, so where the volunteer's shoulder
actually sits relative to the robot base varies session to session,
and the "stressed" centre was recalibrated by eye each time rather
than from a fixed anthropometric offset. Always read the geometry a
given trial actually used from its own `.params.json["path"]`, not
from the placement label.

## Files

One CSV + one JSON sidecar per trial, named
`<subject>_<condition>_<YYYY-MM-DD_HH-MM-SS>.csv` /
`....params.json` (same basename, `.csv` replaced), grouped under
`S01/`, `S02/`, `S03/` — **the placement is not in the filename**, read
it from the sidecar's `path.center`/`path.radius` (nominal =
`[0.45,0,0.45]`/`0.05`, anything else = stressed; see Curation below
for the one-file-per-subject-condition-placement selection already
applied to this deposit). Optional rosbags (full topic recording, see
below) are large and kept separately under `bags/` as one
`.bag.zip`/session (not one per trial — recording spans the whole
session, sliced offline by lap).

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
df = pd.read_csv("S01/S01_E_extended_m4_2026-09-11_12-41-11.csv")
params = json.load(open("S01/S01_E_extended_m4_2026-09-11_12-41-11.params.json"))
placement = "nominal" if (params["path"]["radius"] == 0.05) else "stressed"
laps = df[df["lap"] >= 1]          # drop the training lap
```

The software repository's `matlab/plot_joint_angles_offline.m` and
`matlab/plot_paths_offline.m` reproduce the paper's joint-angle and
traced-path figures directly from a CSV + its sidecar, if MATLAB is
available; they are not required to use the data.

## Curation

The lab machine accumulated many repeat trials per subject/condition
(pilot runs, aborted attempts, and re-runs after recalibration) spread
across 2026-08-28 through 2026-09-11. This deposit keeps **exactly one
trial per `(subject, condition, placement)` cell**, selected as:

1. Drop any trial with no `.params.json` sidecar — these predate
   commit `1d0c4f9` (2026-09-09) and used the buggy pre-fix q3/q4
   convention and/or an ungated `joint_safety` formula (27 trials
   dropped this way; see `docs/paper_update_2026-09-07_to_09.md` in the
   software repo). This incidentally also drops 2 `E_extended_m4`
   trials on intermediate commit `a61ffdd` (joint_safety gated but not
   yet sign-corrected) for S02, since a later `1d0c4f9` trial existed
   in the same cell.
2. Of what's left, keep the chronologically **latest** trial that
   completed at least one full lap (a 0-lap file is an aborted
   attempt, not a usable "last trial" — the single case this mattered,
   `S02_A_standalone` nominal, would otherwise have kept a 12 s run
   with 86 % of samples outside the subject's calibrated joint range).

Result: 30 trials (10 per subject: A/B/C/D/E × nominal/stressed), all
on commit `1d0c4f9`, all 7 laps. `MANIFEST.csv` records which file was
kept for each cell and why (`collected` / `notes` columns); the
superseded repeats and pre-fix files are not included in this deposit.

**Condition F is not in this deposit.** `baseline_aan_node.py` never
writes a `.params.json` sidecar (by design — it logs a reduced topic
set, see the software README), so every F trial was dropped by rule 1
above. Decide before publishing: either add sidecar support to
`baseline_aan_node.py` and re-export, or document F as a planned
addition and publish A–E now.

Two stray desktop screenshots (`Captura de pantalla ....png`,
`gripper_grasping.png`) were present in the raw export and are not
included here — not experiment data.

## Known limitations / caveats

- Gestures/forces are not matched across trials or subjects (natural
  tracing, not a scripted force profile) — expect `|f|` to vary trial
  to trial.
- The manipulability factor (`manip_h`, `w_qr`) penalises
  *degradation* of the robot's manipulability index, not low absolute
  values — it does not, by design, prevent the FR3 approaching its own
  reach/singularity boundary in the stressed placement.
- The stressed-placement geometry (`path.center`) varies across
  sessions for the same subject (chair height was not standardised —
  see "Placement" above); compare `eta`/`rho` trends within a trial
  against that trial's own sidecar geometry, not against a nominal
  "the stressed condition" assumption.
- A handful of kept trials still show `|rho_i| > 1` (the joint reading
  briefly outside that subject's calibrated range) for a small
  fraction of samples — expected occasionally near the edge of a ROM
  sweep that under-covers the task; largest cases are
  `S02_D_manip_m3` stressed and a few `S03_C_jointsafety_m3` trials
  (up to ~20 % of samples). Not a convention bug (nothing near the old
  pre-fix ~1.4 magnitude), just a calibration-tightness note.

## License

Data: CC-BY 4.0 (proposed — confirm before publishing). Software:
MIT, Telerobotic and Interactive Systems Laboratory (TaISLab),
University of Málaga — see the software repository's `LICENSE`.

## Citation

*(add the paper citation once accepted/published)*

## Contact

J. Manuel Gómez-de-Gabriel — TaISLab, University of Málaga —
jesus.gomez@uma.es
