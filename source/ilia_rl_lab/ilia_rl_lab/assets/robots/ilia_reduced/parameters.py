"""Simulator-independent parameters for the reduced ILIA robot.

Copied from unitree_rl_mjlab at d45a7317d1a0abcd996719514f51a48d9fc046ca,
src/assets/robots/ilia_reduced/ilia_reduced_constants.py.
"""

from dataclasses import dataclass


# Canonical control order. Simulator joint indices must be mapped to this order.
ILIA_REDUCED_DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


@dataclass(frozen=True)
class ActuatorParameters:
    """Joint-side controller and physical parameters; all angles use radians."""

    kp: float                 # Nm/rad
    kd: float                 # Nm*s/rad: controller derivative gain
    effort_limit: float       # Nm: nominal limit, before derating
    armature: float           # kg*m^2: added joint-side inertia
    dry_friction: float       # Nm: source model's dry-friction magnitude
    viscous_friction: float   # Nm*s/rad: physical joint damping


# Match the source's waist prior for the waist and every arm joint.
_WAIST_PARAMETERS = ActuatorParameters(
    kp=120.0,
    kd=1.5,
    effort_limit=33.6,
    armature=0.008934561294513125,
    dry_friction=0.0,
    viscous_friction=2.1534676738430187,
)

# Positional arguments follow the six fields declared above.
ILIA_REDUCED_ACTUATOR_PARAMETERS = {
    "left_hip_pitch_joint": ActuatorParameters(
        130.0, 1.5, 72.0, 0.01783925414737667, 0.0, 2.9250907671674806
    ),
    "left_hip_roll_joint": ActuatorParameters(
        120.0, 1.5, 33.6, 0.008934561294513125, 0.14771473587298156, 2.4309900243790903
    ),
    "left_hip_yaw_joint": ActuatorParameters(
        120.0, 1.5, 33.6, 0.008934561294513125, 0.0, 0.0
    ),
    "left_knee_joint": ActuatorParameters(
        130.0, 1.5, 144.0, 0.07135701658950667, 0.8795828150049513, 0.23882387750811213
    ),
    "left_ankle_pitch_joint": ActuatorParameters(
        120.0, 1.5, 67.2, 0.0357382451780525, 1.448989853181378, 0.17986954170462324
    ),
    "right_hip_pitch_joint": ActuatorParameters(
        130.0, 1.5, 72.0, 0.01783925414737667, 0.9558653640939823, 0.7656926584718584
    ),
    "right_hip_roll_joint": ActuatorParameters(
        120.0, 1.5, 33.6, 0.008934561294513125, 0.28975313393307806, 0.0
    ),
    "right_hip_yaw_joint": ActuatorParameters(
        120.0, 1.5, 33.6, 0.008934561294513125, 0.6604591838529709, 3.74048448636575
    ),
    "right_knee_joint": ActuatorParameters(
        130.0, 1.5, 144.0, 0.07135701658950667, 0.9020059524784924, 0.0
    ),
    "right_ankle_pitch_joint": ActuatorParameters(
        120.0, 1.5, 67.2, 0.0357382451780525, 0.6824785374753993, 0.29839634731032993
    ),
    "waist_yaw_joint": _WAIST_PARAMETERS,
    "left_shoulder_pitch_joint": _WAIST_PARAMETERS,
    "left_shoulder_roll_joint": _WAIST_PARAMETERS,
    "left_shoulder_yaw_joint": _WAIST_PARAMETERS,
    "left_elbow_joint": _WAIST_PARAMETERS,
    "right_shoulder_pitch_joint": _WAIST_PARAMETERS,
    "right_shoulder_roll_joint": _WAIST_PARAMETERS,
    "right_shoulder_yaw_joint": _WAIST_PARAMETERS,
    "right_elbow_joint": _WAIST_PARAMETERS,
}


ILIA_REDUCED_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.174,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.087,
    "left_knee_joint": 0.349,
    "left_ankle_pitch_joint": -0.191,
    "right_hip_pitch_joint": -0.174,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.087,
    "right_knee_joint": 0.349,
    "right_ankle_pitch_joint": -0.191,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.261,
    "left_shoulder_roll_joint": 0.174,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.134,
    "right_shoulder_pitch_joint": 0.261,
    "right_shoulder_roll_joint": -0.174,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -1.134,
}

ILIA_REDUCED_INITIAL_ROOT_POS = (0.0, 0.0, 0.72)  # metres
ILIA_REDUCED_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Torque caps use derated effort; action scales retain nominal effort.
EFFORT_DERATE = 0.95
ILIA_REDUCED_EFFORT_LIMITS = {
    name: EFFORT_DERATE * parameters.effort_limit
    for name, parameters in ILIA_REDUCED_ACTUATOR_PARAMETERS.items()
}
ILIA_REDUCED_ACTION_SCALE = {
    name: 0.25 * parameters.effort_limit / parameters.kp
    for name, parameters in ILIA_REDUCED_ACTUATOR_PARAMETERS.items()
}

# Fail early if future edits break the common control order.
if tuple(ILIA_REDUCED_ACTUATOR_PARAMETERS) != ILIA_REDUCED_DOF_NAMES:
    raise ValueError("Actuator parameters must follow the canonical joint order.")
if tuple(ILIA_REDUCED_DEFAULT_JOINT_POS) != ILIA_REDUCED_DOF_NAMES:
    raise ValueError("Default joint positions must follow the canonical joint order.")
