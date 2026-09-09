# Joint-limit-safety factor `η_{k3}` — method notes

Reference for the paper. Final form as of commit `efade57`
(`src/sc_ros_empathic/performance.py :: joint_safety_factor`). The
running controller uses exactly this function
(`shared_control_core.py` → `candidate_performance`); the surface figure
`matlab/plot_joint_safety_surface.m` re-implements the same formula for
plotting.

## Purpose

For each candidate command $v_k$ ($k \in \{h, r, s\}$ — human, robot,
blend), $\eta_{k3} \in (0, 1]$ attenuates that command's contribution
to the shared velocity if it would push the estimated human arm joints
towards their physiological limits. The human arm is the 4-DoF model
(shoulder flex/ext, abd/add, int/ext rotation, elbow flexion);
$l_1, l_2$ are the live upper-arm / forearm lengths from the
visuo-tactile pipeline.

## Formulation

For each joint $i \in \{1,\dots,4\}$ with pre-registered limits
$[q_{\min,i}, q_{\max,i}]$:

$$
\rho_i = \frac{q_i - q_{\mathrm{mid},i}}{q_{\mathrm{half},i}}
\in [-1, 1], \qquad
\mathrm{margin}_i = 1 - |\rho_i|
$$

$$
w_{\mathrm{prox},i} =
\operatorname{clip}\!\left(\frac{\tau - \mathrm{margin}_i}{\tau},\, 0,\, 1\right)
$$

$$
\dot q_i = \big[\, J_{\mathrm{arm}}(q, l_1, l_2)^{+}\, v_k \,\big]_i ,
\qquad
s_i = \dot q_i \cdot \operatorname{sign}(\rho_i)
$$

$$
\mathrm{penalty}_i = \max\!\Big(0,\;
w_s\, \tau\, w_{\mathrm{prox},i}
\;+\; w_d\, w_{\mathrm{prox},i}\,
\frac{s_i}{\max(\mathrm{margin}_i,\, m_{\mathrm{floor}})}
\Big)
$$

$$
\boxed{\;\eta_{k3} = \exp\!\Big(-C_s \textstyle\sum_i \mathrm{penalty}_i\Big)\;}
$$

- $J_{\mathrm{arm}}^{+}$ is the damped least-squares pseudo-inverse of
  the 4-DoF human-arm Jacobian (damping $10^{-3}$). It is computed once
  per control cycle and reused for $v_h$, $v_r$ and $\hat v_s$.
- $s_i$ is the **signed** rate towards the nearer limit: $s_i > 0$
  approaching, $s_i < 0$ retreating.
- $s_i / \mathrm{margin}_i \equiv 1 / (\text{time-to-limit})_i$.

### Parameters (`config/shared_control.yaml`)

| symbol | param | value | meaning |
|---|---|---|---|
| $C_s$ | `Cs` | 12 | penalty exponential gain |
| $\tau$ | `proximity_threshold` | 0.3 | caution band: $w_{\mathrm{prox}} > 0$ once $\lvert\rho_i\rvert > 1-\tau$ |
| $w_s$ | `static_weight` | 0.5 | static (position) term weight |
| $w_d$ | `dynamic_weight` | 0.5 | dynamic (velocity) term weight |
| $m_{\mathrm{floor}}$ | `margin_floor` | 0.05 | floor on $1/\mathrm{margin}$ amplification |

Blend weight in the overall efficiency: `w_joint_safety` = 16 (of
`1, 1, 16, 24` for smoothness / directness / joint_safety /
manipulability).

## Design rationale

1. **Two terms — static (position) + dynamic (velocity).** The human-arm
   Jacobian can have a near-zero-norm column for some joint at some
   posture (a pure axial rotation of a link does not instantly translate
   the wrist), so a purely dynamic term would be blind to a joint that
   is already at risk. The static term is an artificial-potential-style
   backstop.

2. **Proximity gate $w_{\mathrm{prox},i}$.** Both terms are zero while
   the joint sits within $1-\tau$ of mid-range. A brisk mid-range motion
   is not a joint-limit-safety concern; without the gate the dynamic
   term fired on *any* motion and, near an arm-Jacobian singularity
   (large $\dot q$), collapsed $\eta_{k3}$ far from any limit.

3. **Signed dynamic term ($s_i$, not $\max(0, s_i)$) — "relief credit".**
   It penalises an approach and *credits* a retreat: near a limit, a
   command that moves the joint away offsets the static term and
   $\eta_{k3}$ recovers towards 1. The factor does not resist the human
   doing the safe thing. (Earlier one-sided version stuck at the static
   floor $\approx 0.15$–$0.4$ regardless of command direction.)

4. **Per-joint floor $\max(0, \cdot)$.** One joint retreating cannot
   license another joint approaching its limit.

5. **$s_i / \mathrm{margin}_i = 1/(\text{time-to-limit})$.** The penalty
   scales with *both* the approach speed and the current closeness; a
   fast approach far from the limit is treated like a slow approach near
   it.

6. **$m_{\mathrm{floor}}$** bounds the $1/\mathrm{margin}$ amplification
   at the limit itself.

### Resulting shape

`matlab/plot_joint_safety_surface.m` — axes $q_i \times \dot q_i \times
\eta_{k3}$ (see `docs/joint-limit safety factor.pdf`):

- flat plateau at $\eta_{k3} = 1$ outside the caution band, for any
  velocity;
- a valley falling to 0 for a fast approach to a limit;
- a slope rising **back to 1** for a retreat.

## Joint-angle convention fixes (affect what $\rho_i$ means)

The visuo-tactile pipeline's `/right_arm/joint_states` conventions do
not match the DH model; the node reconciles them at the single input
point (`_human_joint_state_cb`) so the control law, the arm Jacobian
and every plot agree:

- **q4 (elbow).** Pipeline sends the elbow **interior angle**
  ($\pi$ rad = arm extended, decreasing with flexion). Node applies
  $q_4 \leftarrow \pi - q_{4,\text{pipeline}}$ → elbow flexion from the
  extended arm ($0$ = straight), matching `DEFAULT_JOINT_LIMITS`
  $q_4 \in [0, 2.53]$ and the DH term $\theta_4 = \pi/2 - q_4$.
- **q3 (shoulder int/ext rotation).** The model's $q_3 = 0$ is neutral
  rotation (forearm in the sagittal plane, verified by FK); the pipeline
  reads $\approx +0.87$ rad there. Subtracted via
  `human_joint_offsets: [0, 0, 0.875, 0]` (a per-pipeline residual, not
  per-volunteer). Refine from a direct neutral-posture reading of
  `position[3]`.

Residual per-joint fixups use the affine map
$q_{\mathrm{model}} = \text{gain} \cdot (q - \text{offset})$
(`~human_joint_offsets` / `~human_joint_gains`, default identity).

## Traceability

| commit | change |
|---|---|
| `a61ffdd` | proximity gate on the dynamic term (was ungated → fired on any motion) |
| `0408aa1` | signed dynamic term + per-joint floor (relief credit for retreating) |
| `9d510fa` | q3 pipeline zero offset (`human_joint_offsets[2] = 0.875`) |
| `e831eab` | q4 interior-angle → flexion conversion at the node input |
| `acf4dd6` | `matlab/plot_joint_safety_surface.m` (method figure) |
| `efade57` | `docs/joint-limit safety factor.pdf` (exported figure) |
