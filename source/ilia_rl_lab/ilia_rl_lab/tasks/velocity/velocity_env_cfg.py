"""Shared manager-based velocity task."""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from . import mdp
from .terrains import ROUGH_TERRAINS_CFG


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    terrain_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1, size=(1.6, 1.0), ordering="xy"
        ),
        max_distance=5.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0, color=(0.85, 0.85, 0.85)
        ),
    )


@configclass
class CommandsCfg:
    twist = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.4, 0.4),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.OrderedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ActorObservationsCfg(ObsGroup):
    base_ang_vel = ObsTerm(
        func=mdp.root_angular_velocity,
        noise=UniformNoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=UniformNoise(n_min=-0.05, n_max=0.05),
    )
    command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "twist"}
    )
    phase = ObsTerm(func=mdp.gait_phase)
    joint_pos = ObsTerm(
        func=mdp.joint_position_error,
        noise=UniformNoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_velocity,
        noise=UniformNoise(n_min=-1.5, n_max=1.5),
    )
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        noise=UniformNoise(n_min=-0.1, n_max=0.1),
        scale=0.2,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticObservationsCfg(ActorObservationsCfg):
    base_lin_vel = ObsTerm(func=mdp.root_linear_velocity)
    foot_height = ObsTerm(func=mdp.foot_height)
    foot_air_time = ObsTerm(func=mdp.foot_air_time)
    foot_contact = ObsTerm(func=mdp.foot_contact)
    foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces)

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    actor: ActorObservationsCfg = ActorObservationsCfg()
    critic: CriticObservationsCfg = CriticObservationsCfg()


@configclass
class EventsCfg:
    # If enabled for play, choose an origin BEFORE resetting the robot.
    randomize_terrain: EventTerm | None = None

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    foot_material = EventTerm(
        func=mdp.SharedFootMaterial,
        mode="startup",
        params={"mu_range": (0.3, 1.6), "num_buckets": 64},
    )

    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "com_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_root_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


@configclass
class RewardsCfg:
    track_linear_velocity = RewTerm(
        func=mdp.track_linear_velocity, weight=1.0
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_angular_velocity, weight=1.0
    )
    body_orientation_l2 = RewTerm(
        func=mdp.torso_orientation_l2, weight=-1.0
    )
    pose = RewTerm(func=mdp.variable_posture, weight=1.0)
    body_ang_vel = RewTerm(
        func=mdp.torso_angular_velocity_l2, weight=-0.05
    )
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.025)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_acc_l2 = RewTerm(func=mdp.action_acc_l2, weight=-0.02)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    foot_gait = RewTerm(func=mdp.feet_gait, weight=0.5)
    foot_clearance = RewTerm(func=mdp.feet_clearance, weight=-1.0)
    foot_swing_height = RewTerm(func=mdp.FeetSwingHeight, weight=-2.0)
    foot_slip = RewTerm(func=mdp.feet_slip, weight=-0.25)
    soft_landing = RewTerm(func=mdp.soft_landing, weight=-1.0e-3)
    stand_still = RewTerm(func=mdp.stand_still, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    illegal_contact = DoneTerm(func=mdp.illegal_contact)


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_curriculum)
    command_vel = CurrTerm(func=mdp.command_curriculum)


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: VelocitySceneCfg = VelocitySceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        # Preserve contact history at each physics step.
        lazy_sensor_update=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    foot_body_names: list[str] = MISSING
    torso_body_name: str = MISSING
    sole_offset: tuple[float, float, float] = MISSING
    posture_std_walking: list[float] = MISSING
    posture_std_running: list[float] = MISSING
    velocity_stages: list[dict] = MISSING

    posture_std_standing: float = 0.05
    gait_period: float = 0.6
    stance_fraction: float = 0.5
    target_foot_height: float = 0.1

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain_scan.update_period = self.sim.dt * self.decimation

        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
