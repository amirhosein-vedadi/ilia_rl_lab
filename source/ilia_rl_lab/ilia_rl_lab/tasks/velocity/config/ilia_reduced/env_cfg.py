"""ILIA Reduced specializations of the shared velocity task."""

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ilia_rl_lab.assets.robots.ilia_reduced.parameters import (
    ILIA_REDUCED_ACTION_SCALE,
    ILIA_REDUCED_DOF_NAMES,
)
from ilia_rl_lab.assets.robots.ilia_reduced.robot_cfg import (
    get_ilia_reduced_robot_cfg,
)

from ... import mdp
from ...velocity_env_cfg import VelocityEnvCfg


@configclass
class IliaReducedRoughEnvCfg(VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = get_ilia_reduced_robot_cfg()
        self.actions.joint_pos.joint_names = list(ILIA_REDUCED_DOF_NAMES)
        self.actions.joint_pos.scale = dict(ILIA_REDUCED_ACTION_SCALE)

        self.foot_body_names = ["left_foot_link", "right_foot_link"]
        self.torso_body_name = "torso_link"
        self.sole_offset = (0.0, 0.0, -0.058)
        self.events.torso_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.torso_body_name]
        )

        self.gait_period = 0.8
        self.stance_fraction = 0.5
        self.target_foot_height = 0.09

        # Same order as ILIA_REDUCED_DOF_NAMES.
        self.posture_std_walking = (
            [0.5, 0.12, 0.12, 0.5, 0.18] * 2
            + [0.12]
            + [0.15, 0.1, 0.1, 0.1] * 2
        )
        self.posture_std_running = (
            [0.6, 0.2, 0.2, 0.6, 0.25] * 2
            + [0.2]
            + [0.25, 0.15, 0.15, 0.15] * 2
        )

        ranges = self.commands.twist.ranges
        ranges.lin_vel_x = (-0.3, 0.4)
        ranges.lin_vel_y = (-0.1, 0.1)
        ranges.ang_vel_z = (-0.2, 0.2)

        self.velocity_stages = [
            {
                "step": 4000 * 24,
                "lin_vel_x": (-0.4, 0.6),
                "lin_vel_y": (-0.15, 0.15),
                "ang_vel_z": (-0.25, 0.25),
            },
            {
                "step": 8000 * 24,
                "lin_vel_x": (-0.5, 0.65),
                "lin_vel_y": (-0.2, 0.2),
                "ang_vel_z": (-0.3, 0.3),
            },
            {
                "step": 12000 * 24,
                "lin_vel_x": (-0.5, 0.7),
                "lin_vel_y": (-0.3, 0.3),
                "ang_vel_z": (-0.4, 0.4),
            },
        ]


@configclass
class IliaReducedFlatEnvCfg(IliaReducedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain_scan = None
        self.observations.actor.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


def configure_play(cfg):
    cfg.scene.num_envs = 32
    cfg.episode_length_s = 1.0e9
    cfg.observations.actor.enable_corruption = False
    cfg.events.push_robot = None
    cfg.curriculum.command_vel = None
    cfg.curriculum.terrain_levels = None

    cfg.commands.twist.ranges.lin_vel_x = (-0.5, 1.0)
    cfg.commands.twist.ranges.lin_vel_y = (-0.3, 0.3)
    cfg.commands.twist.ranges.ang_vel_z = (-0.4, 0.4)

    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.curriculum = False
        generator.num_rows = 5
        generator.num_cols = 5
        generator.border_width = 10.0
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.events.randomize_terrain = EventTermCfg(
            func=mdp.randomize_terrain, mode="reset"
        )


@configclass
class IliaReducedRoughPlayEnvCfg(IliaReducedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class IliaReducedFlatPlayEnvCfg(IliaReducedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
