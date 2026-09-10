# Controller revision, 2026-09-07 – 09-09 — notes for the paper update

Scope: everything changed in `sc_ros_empathic` in this window (commits
`e831eab` .. `1d0c4f9`, full list at the end). All of it concerns the
**human joint-limit-safety pathway** — what `q_i` means, and the
`joint_safety` factor formula itself — triggered by building the live +
offline verification tooling (Sec. "New tooling" below) and reading real
trials through it. The path-following / smoothness / directness /
manipulability machinery is unchanged.

For the full formula and design rationale of the factor in its final
form, see `docs/joint_safety_factor.md` and the figure
`docs/joint-limit safety factor.pdf`; this note is the narrative of
*what was wrong, what changed, and what it moves in the results* — for
the revision text / response-to-reviewers / changelog side of the paper
rather than the method equations themselves.

## 1. Motivation

Building `plot_joint_angles.py` (live) and `plot_joint_angles_offline.m`
(from trial CSVs) to *verify* the human joint-safety pipeline surfaced
three problems that were invisible in aggregate metrics alone:

- q4 (elbow) moved the **wrong way** on screen: extending the arm made
  the plotted angle rise instead of fall.
- q3 (shoulder internal/external rotation) sat pinned near its declared
  limit for almost the entire nominal-placement task.
- `joint_safety` collapsed towards 0 **with every joint comfortably
  mid-range**, and, once a joint was in its caution band, stayed
  attenuated even while the participant was visibly moving it *away*
  from the limit.

## 2. What the joint angles were measuring (convention fixes)

The 4-DoF human-arm model (`DEFAULT_JOINT_LIMITS`, the DH chain in
`dh_utils.py`) assumes `q_i = 0` at the goniometric neutral for each
joint, with q4 (elbow) increasing with flexion from the extended arm.
The visuo-tactile pipeline's `/right_arm/joint_states` does not use
that convention for two of the four joints:

- **q4.** The pipeline reports the elbow's **interior angle**
  (`pi` rad = fully extended, decreasing towards `~0.6` rad fully
  flexed) — the *opposite* sign convention from the model. Fixed at the
  single point of entry (`_human_joint_state_cb`):
  `q4 <- pi - q4_pipeline` (commit `e831eab`).
- **q3.** The DH model's `q3 = 0` is neutral rotation (forearm in the
  sagittal plane — confirmed by forward-kinematics inspection: at
  `q3=0` the forearm direction has no lateral component for any
  shoulder posture). The pipeline reads `q3 ~= +0.87` rad at that same
  neutral posture. A per-joint affine calibration mechanism
  (`~human_joint_offsets` / `~human_joint_gains`,
  `q_model = gain*(q_pipeline - offset)`) already existed for exactly
  this; `human_joint_offsets = [0, 0, 0.875, 0]` is now the shipped
  default in `config/shared_control.yaml` (commit `9d510fa`).

Both corrections apply before the joint-safety factor, the arm
Jacobian, and every plot/log see `q_h` — so the control law, the
Jacobian-based term, and the visualisations are all consistent by
construction (single point of entry, no duplicated logic).

**Effect on the reported quantity `rho_i` (normalised joint position,
what the paper's joint-margin discussion is built on):** before the q3
fix, a participant's shoulder rotation during the ordinary nominal task
read as `rho_3 ~= 0.75-0.85`, regularly **exceeding 1** (i.e. reading
past the declared physiological limit) with observed values up to
`rho_3 ~= 1.4`. After the fix, the same posture reads mid-band
(`rho_3` around 0.2-0.4 at rest). This was a measurement artefact, not
a change in what the participant was doing.

## 3. The `joint_safety` factor formula

Two changes to `performance.joint_safety_factor`, both about the
**dynamic (closing-rate) term** — the static (position-only) term is
unchanged.

### 3.1 Proximity gate (commit `a61ffdd`)

Before: the dynamic term `sum_i qdot_i / margin_i` was evaluated for
*every* joint regardless of position. With `margin_i ~= 0.7-1.0` for a
comfortably mid-range joint, the division barely damps it, so any
brisk hand motion accumulated penalty — and near a human-arm-Jacobian
near-singularity (an inherent feature of the redundant 4-DoF arm model)
`qdot` spikes drove `joint_safety` towards 0 with every joint far from
any limit.

After: both the static and dynamic terms are multiplied by the same
proximity weight `prox_w_i = clip((tau - margin_i)/tau, 0, 1)`, which
is exactly 0 while `|rho_i| < 1 - tau`. A joint sitting mid-range no
longer lowers `joint_safety`, however fast the candidate command moves
it; the factor only engages once a joint enters the outer
`proximity_threshold` (`tau = 0.3`) band of its declared range.

### 3.2 Signed dynamic term / "relief credit" (commit `0408aa1`)

Before (even after 3.1): once a joint was inside the caution band, the
static term produced a floor of roughly `exp(-Cs * static_weight * tau)
~= 0.15` regardless of the candidate command's direction — a retreating
motion was attenuated by exactly as much as an approaching one.

After: the closing-rate term keeps its sign
(`closing_rate_i = qdot_i * sign(rho_i)`, positive when approaching the
nearer limit, negative when retreating), and each joint's contribution
to the penalty is floored at 0 individually:

```
penalty_i = max(0,  static_weight * tau * prox_w_i
                   + dynamic_weight * prox_w_i * closing_rate_i
                                      / max(margin_i, margin_floor) )
joint_safety = exp(-Cs * sum_i penalty_i)
```

A command that moves an at-limit joint away now earns a credit that
offsets the static term, and `joint_safety` recovers towards 1; an
approach is unaffected (still collapses towards 0). The per-joint
`max(0, .)` keeps one joint's retreat from licensing another joint's
approach.

`closing_rate_i / margin_i` is exactly `1 / (time-to-limit)`, so the
penalty already scaled with both approach speed and proximity before
this change; 3.2 only changes the sign convention and the floor.

## 4. New tooling (supporting infrastructure, not a method change)

- `scripts/joint_range_probe.py` — records live per-joint min/max
  through the same conversions the controller applies, prints a
  paste-ready `joint_limits_rad:` block for a volunteer's subject file
  (commits `425ab6c`, `d579071`).
- Live (`plot_joint_angles.py`) and offline
  (`matlab/plot_joint_angles_offline.m`) viewers: per-joint subplot
  with the declared limits **and** the caution-band lines drawn
  (`rho = +-(1-tau)`, commit `7a736e3`), plus a subplot with
  `eta_h/eta_r/eta_s` and their `joint_safety` component together
  (commit `ef17340`).
- `matlab/plot_joint_safety_surface.m` + `docs/joint-limit safety
  factor.pdf` — the eta_k3(q, qdot) method figure (commits `acf4dd6`,
  `efade57`).
- `docs/joint_safety_factor.md` — the formula/rationale reference
  (commit `1d0c4f9`).

## 5. Empirical illustration (pilot, N=1 — not a results-section claim)

Single-participant (S01) pilot pair on the revised controller
(`git 1d0c4f9`), nominal vs. per-participant stressed placement,
condition `E_extended_m4` vs. `B_baseline_m2`, 6 analysed laps/cell
(lap 0 discarded), participant instructed to trace the circle normally
(no deliberate forcing):

| | NOMINAL-E | NOMINAL-B | STRESSED-E | STRESSED-B |
|---|---|---|---|---|
| `\|cross_track\|` med / p90 | 3.3 / 9.3 mm | 5.0 / 11.6 mm | 12.0 / 37.3 mm | 8.9 / 22.7 mm |
| q3 time in caution band | 0% | 0% | 20% | 38% |
| q3 `rho` minimum reached | -0.09 | +0.18 | **+0.11** | **+0.43** |
| robot EE max reach / `w_qr` min | 0.68 m / 0.094 | 0.68 m / 0.093 | 0.856 m / 0.050 | 0.853 m / 0.049 |

Reading: nominal placement gives E ~= B on every metric (no cost when
nothing is at risk, as the framework predicts). In the stressed
placement, E roughly halves the time q3 spends in its caution band and
lets it relax further (`rho_3` down to 0.11 vs. B's floor of 0.43), at
a tracking-accuracy cost (~35% larger cross-track error). The
manipulability factor shows **no** measurable effect on how close the
robot gets to its own reach limit in either condition — expected, since
it penalises *degradation* of `w`, not low absolute `w`, and the path
follower drives the end-effector to that point regardless of the human
(see the "why doesn't `manip` help" discussion, not yet acted on).

Caveats for the paper text: one participant, one trial per cell,
gestures not matched (`|f|` differed 2.7 N vs 6.0 N between the two
nominal runs), q1 was *not* relieved by E in this pair (posture-driven
by the placement, not something the factor can address). This
illustrates the mechanism; it is not powered for the H1-H4 claims.

## 6. Where this likely touches the paper text

- Human-arm model section: state the goniometric-neutral convention
  explicitly and that the pipeline's raw readings require the q3/q4
  conversions above (methods, reproducibility).
- `joint_safety` factor definition (Eq. for eta_k3): update to the
  gated + signed form (Sec. 3 above / `docs/joint_safety_factor.md`).
- Add the eta_k3(q, qdot) surface as a method figure.
- Limitations / discussion: the manipulability factor's
  degradation-only design and its observed lack of effect at the robot's
  reach boundary.
- Any numbers or figures drawn from trials recorded **before**
  `1d0c4f9` are on a different controller and should not be pooled with
  post-fix data.

## Commits in this window

```
e831eab  q4: convert the pipeline's elbow interior angle at the input
425ab6c  Add joint_range_probe.py -- measure the real per-joint ROM
9d510fa  q3: subtract the pipeline's ~+0.87 rad zero offset
a61ffdd  joint_safety: gate the dynamic term by proximity too
7a736e3  viewers: draw the safety margins (rho = +-(1-tau)) as dotted lines
ef17340  viewers: efficiency subplot shows eta_h/eta_r/eta_s + joint_safety
d579071  joint_range_probe: apply the same q transforms as the node
acf4dd6  matlab: eta_k3 surface over (joint angle, joint velocity) for the paper
0408aa1  joint_safety: signed dynamic term (relief credit for retreating)
efade57  docs: add the joint-limit-safety factor surface figure
1d0c4f9  docs: joint_safety_factor.md -- method notes for the paper
```
