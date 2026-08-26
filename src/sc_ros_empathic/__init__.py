"""sc_ros_empathic: reactive, performance-weighted shared control for the
FR3 + compliant gripper pHRI rehabilitation platform.

This package is a straight ROS1/Python port of the offline
simulation/verification code (candidate performance evaluation +
combination law, human-arm kinematics, robot Jacobian access, reactive
path follower). The control law itself is UNCHANGED between simulation
and deployment on purpose, so that anything tuned/validated offline
transfers directly -- see shared_control_core.py's module docstring.
"""
