# sc_ros_empathic

ROS 1 (catkin) implementation of the reactive, performance-weighted
shared-control law for the FR3 + compliant gripper pHRI rehabilitation
platform: combines a human admittance-derived command and a robot
path-following command via per-cycle local-performance weighting,
extended with two novel factors (human joint-limit safety, robot
manipulability/singularity avoidance).

## Layout

- `src/sc_ros_empathic/` -- ROS-independent control-law library
  (ported, logic-unchanged, from the offline simulation/verification
  package this law was tuned in): `shared_control_core.py`,
  `performance.py`, `dh_utils.py` (human-arm kinematics),
  `robot_model.py` + `fr3_model.py` (self-contained FR3 FK + geometric
  Jacobian for the manipulability factor), `path_follower.py`
  (reactive pure-pursuit-style path following), `experiment.py`
  (experimental-condition table, lap counter, joint-margin observable
  -- protocol glue only, no control-law logic), `subject_config.py`
  (per-volunteer YAML loader).
- `scripts/shared_control_node.py` -- main ROS node: reads
  `franka_states` and the human-arm state, runs the shared-control
  law for the selected `~condition`, publishes the Cartesian velocity
  command and the `~diag/*` observables the offline metric analysis
  needs.
- `scripts/baseline_aan_node.py` -- experimental condition F only
  (impedance-control AAN baseline of Zhang et al. [9]); a different
  controller, see its docstring (parameters are placeholders pending
  reconciliation with [9]).
- `scripts/rviz_visualization_node.py` -- publishes the path to trace
  and the current end-effector position as RViz markers.
- `config/shared_control.yaml` -- gains / weights / model parameters
  (PLACEHOLDER pre-hardware values), homogeneous across volunteers,
  loaded by the launch file.
- `config/subjects/<id>.yaml` -- one file per volunteer: id,
  demographics (sex/age/height/mass), arm segment lengths,
  pre-registered joint ranges (see `subject_template.yaml`). Nothing
  experiment-specific.
- `launch/shared_control.launch` -- brings up the FR3's external
  Cartesian velocity controller, the shared-control + RViz nodes, and
  (optionally) a `rosbag record` of the trial.
- `launch/baseline_aan.launch` -- same, for condition F.

## Experimental conditions (paper Sec. V-B)

Select with `condition:=<id>`:

| id                 | m | robot assists | factors                                            |
|--------------------|---|---------------|----------------------------------------------------|
| `A_standalone`     | - | no            | none -- shaped human admittance command only       |
| `B_baseline_m2`    | 2 | yes           | smoothness, directness (baseline reactive SC [1])  |
| `C_jointsafety_m3` | 3 | yes           | baseline + human joint-limit safety (ablation)     |
| `D_manip_m3`       | 3 | yes           | baseline + robot manipulability (ablation)         |
| `E_extended_m4`    | 4 | yes           | all four factors (proposed)                        |
| `F_impedance_aan`  | - | yes           | separate node: `roslaunch sc_ros_empathic baseline_aan.launch` |

An unknown `condition` aborts node startup with the valid list -- there
is no silent default. Each cycle the requested factor set is
intersected with what the sensors can support (fresh `q_h` for
`joint_safety`, a fresh robot Jacobian for `manipulability`); a
downgrade is logged (`logwarn`/`logerr`), never silent. If a condition
that *requires* `manipulability` (D, E) starts with
`~jacobian_source:=none` (or KDL failing to init), the node logs an
error at startup.

## Circle placement (paper Sec. V-A)

Two placements per volunteer, same circle, different location:

- **nominal** -- safe mid-workspace, replicates [1]. Homogeneous
  across volunteers: `path_center:=[0.45, 0.0, 0.45]`, `path_radius:=0.05`.
- **stressed** -- near the edge of the combined human+robot reach
  (~85-90 %), on the line from the robot base through the participant's
  shoulder. Per-participant: pass the calibrated
  `path_center:="[x, y, z]"` for the session.

`placement:=nominal|stressed` is only a label (logging / bag naming);
the geometry is whatever `path_center` / `path_radius` / `path_normal`
you pass. Both are **horizontal circles** by default
(`path_normal:=[0, 0, 1]` -> circle in a constant-z plane), matching
the x-y traced-path plots in the paper (Figs 3, 5, 7). For a
frontal/vertical circle facing the participant, set e.g.
`path_normal:="[1, 0, 0]"` -- the follower and the metrics are
plane-agnostic.

RViz shows the circle **outline** plus a translucent **disc** filling
it, so the plane and its tilt are unambiguous (a bare ellipse outline
gives no depth cue); the red sphere is the live end-effector. Disable
the disc with `show_plane:=false`. `rviz:=true` loads
`rviz/shared_control.rviz` (Fixed Frame `fr3_link0`, MarkerArray on
`/sc_ros_empathic/viz`).

## Per-volunteer configuration

One YAML per volunteer under `config/subjects/` (copy
`subject_template.yaml`). It holds **only human parameters** -- the
volunteer's body, not the experiment: demographics for the participant
table, arm segment lengths, and the pre-registered joint ranges.

```yaml
subject:      { id: S01, date: '2026-09-01', dominant_arm: right,
                sex: f, age_years: 27, height_m: 1.68, mass_kg: 61, notes: '' }
anthropometry:{ l1_m: 0.31, l2_m: 0.27, source: visuo-tactile }
joint_limits_rad:            # OPTIONAL: omit -> study-default ROM
  q1: [-1.05, 3.14]          # shoulder flex/ext
  q2: [0.0, 3.14]            # shoulder abd/add
  q3: [-1.57, 1.57]          # shoulder int/ext rot
  q4: [0.0, 2.53]            # elbow flex/ext
```

Launch with `subject:=S01`; the node resolves
`config/subjects/S01.yaml`. Missing `subject.id` / `l1_m` / `l2_m`, or
a `joint_limits_rad` block that isn't all four joints with `max > min`,
**aborts startup** with a message -- never a silent default. A
`dominant_arm` other than `right` only warns (the visuo-tactile
pipeline tracks the right arm only). Omit `subject:=` to run from plain
params (back-compat).

Everything about the *experiment* is homogeneous across volunteers and
stays in `config/shared_control.yaml`: gains, factor weights, the
joint-margin threshold `tau` (`proximity_threshold`), path geometry,
loop rate. `~human_joint_limits` / `~human_link_lengths` remain as
ad-hoc overrides.

Human-arm state topics (checked against the sibling package
`TaISLab/sc_effort_experiment`, same platform): the 4-DoF joint vector
is `sensor_msgs/JointState` with `position[0:4] = q1..q4` on
`~human_joint_state_topic` -- both packages use the placeholder
`/right_arm/joint_states` and both say "confirm with `rostopic hz`
before every session". Neither the pipeline nor the sibling exposes a
link-lengths topic reliably, so `l1, l2` fall back (topic fresh ->
subject file -> `~human_link_lengths`) and `joint_safety` only drops if
none is available. `q_h` always needs the live topic.

## Recording trials

**Off by default.** No bag is written unless you pass `record:=true`.
Then a `rosbag record` node writes to
`<bag_dir>/<trial_label>_<YYYY-MM-DD-HH-MM-SS>.bag`
(`bag_dir` defaults to `~/sc_ros_empathic_bags`, `trial_label` to
`trial`). The `-o` timestamp means re-runs never overwrite. Recording
starts with the launch and runs for the whole session -- one bag per
session; slice individual trials offline by the `path_progress` lap
index (`>= 1`; lap 0 is the training loop, each trial is four loops).
All A--E conditions publish under the same node name
(`shared_control_node`) so one analysis script handles every bag; F
(`baseline_aan.launch`) records a reduced subset (see below).

Recorded topics (`shared_control.launch`):

| topic | type | contents |
|---|---|---|
| `/franka_state_controller/franka_states` | `franka_msgs/FrankaState` | `q`, `dq`, `tau_J`, `O_T_EE` (EE pose), `O_F_ext_hat_K` (ext. wrench) |
| `/right_arm/joint_states` (`~human_joint_state_topic`) | `sensor_msgs/JointState` | human `q1..q4` from the visuo-tactile pipeline |
| `/right_arm/link_lengths` (`~human_link_lengths_topic`) | `std_msgs/Float64MultiArray` | `[l1, l2]` |
| `/right_arm/arm_points` (`~human_arm_points_topic`) | `std_msgs/Float64MultiArray` (PLACEHOLDER) | shoulder/elbow/wrist points `[sx,sy,sz, ex,ey,ez, wx,wy,wz]` in the pipeline's **fixed** frame -- captures shoulder movement during the trial |
| `/taislab_controller/jacobian` (`~robot_jacobian_topic`) | `std_msgs/Float64MultiArray` (PLACEHOLDER) | FR3 6xN Jacobian, row-major, from the C++ controller (only with `~jacobian_source:=topic`) |
| `/robot_vel_ctrl/vel_cmd` (`~cmd_topic`) | `geometry_msgs/TwistStamped` | emergent command `v_s` sent to the robot |
| `~/eta` | `Float64MultiArray` | `[eta_h, eta_r, eta_s]` |
| `~/diag/condition` | `String` (latched) | condition id |
| `~/diag/factors_layout` | `String` (latched) | `smoothness,directness,joint_safety,manipulability` |
| `~/diag/factors_h`, `~/diag/factors_r` | `Float64MultiArray` | the 4 partial factors for the `v_h` / `v_r` candidates (`NaN` = inactive) |
| `~/diag/v_h`, `~/diag/v_r`, `~/diag/v_s` | `Vector3Stamped` | the three velocity terms |
| `~/diag/force` | `Vector3Stamped` | filtered interaction force driving the admittance |
| `~/diag/joint_margins` | `Float64MultiArray` | `[m1, m2, m3, m4, min]` (only while human state fresh) |
| `~/diag/manipulability` | `Float64` | `w(q_r) = sqrt(det(J J^T))` (only while the Jacobian source is live) |
| `~/diag/path_progress` | `Float64MultiArray` | `[s_near, lap, lap+s_near, cross_track_err_m]` |
| `~/diag/arm_points` | `geometry_msgs/PoseArray` | pipeline shoulder/elbow/wrist re-published (`poses[0..2]`), stamped, frame `~human_points_frame` |
| `~/diag/arm_points_fk` | `geometry_msgs/PoseArray` | shoulder/elbow/wrist from **FK** on `(q_h, l1, l2)`, frame `human_shoulder` (shoulder at origin) -- consistency check vs. the pipeline points; does **not** show shoulder drift |
| `~/diag/arm_points_layout` | `String` (latched) | `shoulder,elbow,wrist` |

Both point triplets are `poses[0]=shoulder, poses[1]=elbow,
poses[2]=wrist`. Shoulder **movement** is only in `arm_points` /
the raw pipeline topic (fixed frame); `arm_points_fk` has the shoulder
pinned at the origin by construction. If the pipeline publishes the
points with a different message type, the raw topic is still recorded
as-is -- only the `~/diag/arm_points` re-publish needs the subscriber
(`_human_arm_points_cb`) adjusted.

Not recorded: TF, RViz markers, `/rosout` (EE pose is already in
`franka_states`). `baseline_aan.launch` records `franka_states`, the
raw `~human_arm_points_topic`, `vel_cmd`, and
`~/diag/{condition,v_h,v_r,v_s,force,path_progress}` -- that node
computes no eta/factors/joint_margins/manipulability/FK.

## Before your first run

Two topics in `shared_control_node.py` are **placeholders pending
confirmation** from whoever owns the visuo-tactile pipeline:

- `~human_joint_state_topic` (default `/right_arm/joint_states`,
  `sensor_msgs/JointState`, expects `position[0:4] = q1..q4`)
- `~human_link_lengths_topic` (default `/right_arm/link_lengths`,
  `std_msgs/Float64MultiArray`, expects `data = [l1, l2]` in meters)
- `~human_arm_points_topic` (default `/right_arm/arm_points`,
  assumed `std_msgs/Float64MultiArray`, `data = [sx,sy,sz, ex,ey,ez,
  wx,wy,wz]` in a fixed frame; set `~human_points_frame` to that
  frame). Recording-only -- the control law does not read it.

If these are wrong, the node does **not** fail loudly -- for a
condition that uses `joint_safety` (C, E) it logs one warning and runs
without that factor; for A, B, D it makes no difference. Run
`rostopic hz <topic>` on both before trusting a C/E session, and
override via launch args once confirmed (no code change needed).

The robot Jacobian (for the manipulability factor) comes from
`~jacobian_source`:

- **`analytic`** (default): `fr3_model.py` computes the FR3 forward
  kinematics and the 6x7 geometric Jacobian in pure numpy from the
  nominal (manufacturer) Denavit-Hartenberg parameters. Self-contained
  -- only input is `q` from `franka_states`; **no PyKDL**, no
  `/robot_description`, no dependency on the velocity controller. The
  lookahead Jacobian `J(q + qdot_candidate*dt)` is evaluated exactly,
  so `eta_k4` is genuinely **per-candidate** as the paper defines it.
  Verified against the known FR3 poses and against a finite-difference
  Jacobian (agreement ~1e-10).
- **`kdl`**: same result via PyKDL + `/robot_description`. Requires
  `python_orocos_kdl`, `kdl_parser_py`, a loaded FR3 URDF, and
  `~base_link` / `~ee_link` (default `fr3_link0` / `fr3_link8`) matching
  it. Only useful if you want the calibrated URDF kinematics rather
  than the nominal ones.
- **`topic`**: a 6xN Jacobian published elsewhere as
  `std_msgs/Float64MultiArray` (flattened **row-major**) on
  `~robot_jacobian_topic` (set `~robot_jacobian_rows` /
  `~robot_jacobian_cols` if not 6x7). Only the *current* J is
  available, so the lookahead is time-propagated (finite difference of
  the last two messages), **not** per-candidate; the factor then
  reduces authority near a singularity rather than ranking candidates.
  Drops on staleness (fail soft).
- **`none`**: manipulability factor disabled.

(`franka_ros` does not publish a Jacobian field -- checked against
`franka_msgs/FrankaState.msg`.)

## Run

Proposed controller, nominal placement, RViz up:

```bash
roslaunch sc_ros_empathic shared_control.launch
```

A recorded stressed-placement trial for volunteer S03, condition E:

```bash
roslaunch sc_ros_empathic shared_control.launch subject:=S03 condition:=E_extended_m4 placement:=stressed path_center:="[0.62, 0.05, 0.30]" record:=true trial_label:=S03_stressed_E
```

Baseline (condition B), nominal:

```bash
roslaunch sc_ros_empathic shared_control.launch subject:=S03 condition:=B_baseline_m2 record:=true trial_label:=S03_nominal_B
```

Condition F (impedance-AAN baseline, separate node):

```bash
roslaunch sc_ros_empathic baseline_aan.launch record:=true trial_label:=P03_nominal_F
```
