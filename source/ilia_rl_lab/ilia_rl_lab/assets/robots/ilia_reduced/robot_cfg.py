"""Isaac Lab articulation configuration for reduced ILIA."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from .actuators import DelayedImplicitActuatorCfg
from .parameters import (
    ILIA_REDUCED_ACTUATOR_PARAMETERS,
    ILIA_REDUCED_DEFAULT_JOINT_POS,
    ILIA_REDUCED_DOF_NAMES,
    ILIA_REDUCED_EFFORT_LIMITS,
    ILIA_REDUCED_INITIAL_ROOT_POS,
    ILIA_REDUCED_SOFT_JOINT_POS_LIMIT_FACTOR,
)

ASSET_DIR = Path(__file__).resolve().parent
URDF_PATH = ASSET_DIR / "urdf" / "ilia_reduced.urdf"


def get_ilia_reduced_robot_cfg() -> ArticulationCfg:
    """Return an independent configuration for each caller."""
    if not URDF_PATH.is_file():
        raise FileNotFoundError(URDF_PATH)

    params = ILIA_REDUCED_ACTUATOR_PARAMETERS
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH),
            usd_dir=str(ASSET_DIR / "generated"),
            usd_file_name="ilia_reduced.usd",
            force_usd_conversion=False,
            fix_base=False,
            root_link_name="base",
            merge_fixed_joints=False,
            make_instanceable=True,
            collision_from_visuals=False,
            collider_type="convex_hull",
            self_collision=False,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ILIA_REDUCED_INITIAL_ROOT_POS,
            joint_pos=dict(ILIA_REDUCED_DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=ILIA_REDUCED_SOFT_JOINT_POS_LIMIT_FACTOR,
        actuators={
            "joints": DelayedImplicitActuatorCfg(
                joint_names_expr=list(ILIA_REDUCED_DOF_NAMES),
                stiffness={name: p.kp for name, p in params.items()},
                damping={name: p.kd for name, p in params.items()},
                effort_limit_sim=dict(ILIA_REDUCED_EFFORT_LIMITS),
                # Effectively unbounded: the mjlab position actuator has no
                # hard velocity brake. This is not a hardware speed rating.
                velocity_limit_sim=1.0e6,
                armature={name: p.armature for name, p in params.items()},
                friction={name: p.dry_friction for name, p in params.items()},
                dynamic_friction={name: p.dry_friction for name, p in params.items()},
                viscous_friction={name: p.viscous_friction for name, p in params.items()},
                min_delay=1,
                max_delay=4,
            ),
        },
    )
