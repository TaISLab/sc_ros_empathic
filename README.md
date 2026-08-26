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
  `robot_model.py` (robot Jacobian via PyKDL), `path_follower.py`
  (reactive pure-pursuit-style path following).
- `scripts/shared_control_node.py` -- main ROS node: reads
  `franka_states` and the human-arm state, runs the shared-control
  law, publishes the Cartesian velocity command.
- `scripts/rviz_visualization_node.py` -- publishes the path to trace
  and the current end-effector position as RViz markers.
- `launch/shared_control.launch` -- brings up the FR3's external
  Cartesian velocity controller, both nodes above, and RViz.

## Before your first run

Two topics in `shared_control_node.py` are **placeholders pending
confirmation** from whoever owns the visuo-tactile pipeline:

- `~human_joint_state_topic` (default `/right_arm/joint_states`,
  `sensor_msgs/JointState`, expects `position[0:4] = q1..q4`)
- `~human_link_lengths_topic` (default `/right_arm/link_lengths`,
  `std_msgs/Float64MultiArray`, expects `data = [l1, l2]` in meters)

If these are wrong, the node does **not** fail loudly -- it logs one
warning and runs without the joint-limit safety factor. Run
`rostopic hz <topic>` on both before trusting a session, and override
via launch args once confirmed (no code change needed).

The robot Jacobian (needed for the manipulability factor) is computed
analytically via PyKDL from `/robot_description` -- **not** from a
`franka_ros`-published Jacobian topic, which does not exist (checked
against the authoritative `franka_msgs/FrankaState.msg`; see
`robot_model.py`'s docstring). Requires `python_orocos_kdl` and
`kdl_parser_py` and a loaded FR3 URDF before this node starts.

`~base_link`/`~ee_link` default to `fr3_link0`/`fr3_link8`: check these
match your actual URDF link names.

## Run

```bash
roslaunch sc_ros_empathic shared_control.launch
```
