"""Velocity-task observations and rewards translated from the mjlab task."""

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def action_term(env):
    return env.action_manager.get_term("joint_pos")


def robot(env):
    return env.scene["robot"]


def command(env):
    return env.command_manager.get_command("twist")


def command_speed(env):
    cmd = command(env)
    return torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()


def indices(env):
    """Cache static name mappings and constant rigid-body parameters."""
    if not hasattr(env, "_velocity_indices"):
        asset = robot(env)
        sensor = env.scene["contact_forces"]

        foot_ids, foot_names = asset.find_bodies(
            env.cfg.foot_body_names, preserve_order=True
        )
        contact_ids, contact_names = sensor.find_bodies(
            env.cfg.foot_body_names, preserve_order=True
        )
        torso_ids, _ = asset.find_bodies(env.cfg.torso_body_name)

        assert foot_names == list(env.cfg.foot_body_names)
        assert contact_names == list(env.cfg.foot_body_names)
        assert len(torso_ids) == 1

        env._velocity_indices = {
            "feet": foot_ids,
            "contact_feet": contact_ids,
            "contact_other": [
                i for i, name in enumerate(sensor.body_names)
                if name not in env.cfg.foot_body_names
            ],
            "torso": torso_ids[0],
            "mass": asset.data.default_mass.to(env.device),
            # Isaac Lab exposes this tensor in the link's actor frame,
            # about the body's center of mass.
            "inertia": asset.data.default_inertia.to(env.device).reshape(
                env.num_envs, asset.num_bodies, 3, 3
            ),
            "walking_std": torch.tensor(
                env.cfg.posture_std_walking, device=env.device
            ),
            "running_std": torch.tensor(
                env.cfg.posture_std_running, device=env.device
            ),
        }
    return env._velocity_indices


def joint_position_error(env):
    data = robot(env).data
    ids = action_term(env).joint_ids
    return data.joint_pos[:, ids] - data.default_joint_pos[:, ids]


def joint_velocity(env):
    data = robot(env).data
    ids = action_term(env).joint_ids
    return data.joint_vel[:, ids] - data.default_joint_vel[:, ids]


def root_linear_velocity(env):
    return robot(env).data.root_link_lin_vel_b


def root_angular_velocity(env):
    return robot(env).data.root_link_ang_vel_b


def gait_phase(env):
    phase = (
        env.episode_length_buf * env.step_dt / env.cfg.gait_period
    ) % 1.0
    result = torch.stack(
        (torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)),
        dim=1,
    )
    standing = torch.linalg.vector_norm(command(env), dim=1) < 0.1
    return torch.where(standing[:, None], torch.zeros_like(result), result)


def height_scan(env):
    """Root-height minus terrain-height; source offset is zero."""
    sensor = env.scene["terrain_scan"]
    heights = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
    return torch.where(
        torch.isfinite(heights), heights, torch.full_like(heights, 5.0)
    )


def foot_state(env):
    """World position and velocity of the two sole points."""
    data = robot(env).data
    ids = indices(env)["feet"]

    local_offset = torch.tensor(
        env.cfg.sole_offset, device=env.device
    ).expand(env.num_envs, len(ids), 3)

    offset_w = quat_apply(data.body_link_quat_w[:, ids], local_offset)
    position = data.body_link_pos_w[:, ids] + offset_w

    # Express point velocity from the body's COM velocity.
    lever = position - data.body_com_pos_w[:, ids]
    velocity = data.body_com_lin_vel_w[:, ids] + torch.linalg.cross(
        data.body_com_ang_vel_w[:, ids], lever, dim=-1
    )
    return position, velocity


def foot_height(env):
    return foot_state(env)[0][..., 2]


def foot_air_time(env):
    sensor = env.scene["contact_forces"]
    return sensor.data.current_air_time[:, indices(env)["contact_feet"]]


def feet_in_contact(env):
    sensor = env.scene["contact_forces"]
    ids = indices(env)["contact_feet"]
    return sensor.data.current_contact_time[:, ids] > 0.0


def foot_contact(env):
    return feet_in_contact(env).float()


def foot_forces(env):
    sensor = env.scene["contact_forces"]
    return sensor.data.net_forces_w[:, indices(env)["contact_feet"]]


def foot_contact_forces(env):
    forces = foot_forces(env).flatten(start_dim=1)
    return torch.sign(forces) * torch.log1p(forces.abs())


def first_foot_contact(env):
    sensor = env.scene["contact_forces"]
    return sensor.compute_first_contact(env.step_dt)[
        :, indices(env)["contact_feet"]
    ]


def track_linear_velocity(env):
    actual = root_linear_velocity(env)
    error = (command(env)[:, :2] - actual[:, :2]).square().sum(dim=1)
    error += 2.0 * actual[:, 2].square()
    return torch.exp(-error / 0.25)


def track_angular_velocity(env):
    actual = root_angular_velocity(env)
    error = (command(env)[:, 2] - actual[:, 2]).square()
    error += 0.05 * actual[:, :2].square().sum(dim=1)
    return torch.exp(-error / 0.5)


def torso_orientation_l2(env):
    data = robot(env).data
    quat = data.body_link_quat_w[:, indices(env)["torso"]]
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    projected = quat_apply_inverse(quat, gravity)
    return projected[:, :2].square().sum(dim=1)


def torso_angular_velocity_l2(env):
    data = robot(env).data
    velocity = data.body_com_ang_vel_w[:, indices(env)["torso"], :2]
    return velocity.square().sum(dim=1)


def variable_posture(env):
    speed = command_speed(env)[:, None]
    mapping = indices(env)
    walking = mapping["walking_std"][None, :]
    running = mapping["running_std"][None, :]
    standing = torch.full_like(walking, env.cfg.posture_std_standing)

    std = torch.where(
        speed < 0.1, standing,
        torch.where(speed < 1.5, walking, running),
    )
    return torch.exp(
        -(joint_position_error(env).square() / std.square()).mean(dim=1)
    )


def angular_momentum_l2(env):
    data = robot(env).data
    mapping = indices(env)
    mass = mapping["mass"]
    inertia = mapping["inertia"]

    position = data.body_com_pos_w
    velocity = data.body_com_lin_vel_w
    omega_w = data.body_com_ang_vel_w
    quat = data.body_link_quat_w

    total_mass = mass.sum(dim=1, keepdim=True)
    com = (mass[..., None] * position).sum(dim=1) / total_mass
    com_velocity = (mass[..., None] * velocity).sum(dim=1) / total_mass

    omega_b = quat_apply_inverse(quat, omega_w)
    spin_b = torch.matmul(inertia, omega_b.unsqueeze(-1)).squeeze(-1)
    spin_w = quat_apply(quat, spin_b)

    orbital = torch.linalg.cross(
        position - com[:, None, :],
        mass[..., None] * (velocity - com_velocity[:, None, :]),
        dim=-1,
    )
    momentum = (spin_w + orbital).sum(dim=1)
    env.extras.setdefault("log", {})["Metrics/angular_momentum_mean"] = (
        torch.linalg.vector_norm(momentum, dim=1).mean()
    )
    return momentum.square().sum(dim=1)


def action_acc_l2(env):
    term = action_term(env)
    acceleration = (
        term.raw_actions - 2.0 * term.previous + term.previous_previous
    )
    env.extras.setdefault("log", {})["Metrics/mean_action_acc"] = (
        acceleration.abs().mean()
    )
    return acceleration.square().sum(dim=1)


def feet_gait(env):
    phase = (
        env.episode_length_buf * env.step_dt / env.cfg.gait_period
    )[:, None]
    offsets = torch.tensor([0.0, 0.5], device=env.device)[None, :]
    expected_stance = ((phase + offsets) % 1.0) < env.cfg.stance_fraction
    reward = (expected_stance == feet_in_contact(env)).float().mean(dim=1)
    return reward * (command_speed(env) > 0.1)


def feet_clearance(env):
    position, velocity = foot_state(env)
    speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1)
    error = (position[..., 2] - env.cfg.target_foot_height).abs()
    return (error * speed).sum(dim=1) * (command_speed(env) > 0.1)


class FeetSwingHeight(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.peak = torch.zeros(
            (env.num_envs, len(env.cfg.foot_body_names)), device=env.device
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.peak[ids] = 0.0

    def __call__(self, env):
        height = foot_height(env)
        self.peak = torch.where(
            ~feet_in_contact(env), torch.maximum(self.peak, height), self.peak
        )
        landing = first_foot_contact(env)
        error = self.peak / env.cfg.target_foot_height - 1.0
        cost = (error.square() * landing).sum(dim=1)
        cost *= command_speed(env) > 0.1

        env.extras.setdefault("log", {})["Metrics/peak_height_mean"] = (
            (self.peak * landing).sum() / landing.sum().clamp_min(1)
        )
        self.peak = torch.where(landing, torch.zeros_like(self.peak), self.peak)
        return cost


def feet_slip(env):
    velocity = foot_state(env)[1][..., :2]
    contact = feet_in_contact(env)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    env.extras.setdefault("log", {})["Metrics/slip_velocity_mean"] = (
        (speed * contact).sum() / contact.sum().clamp_min(1)
    )
    return (
        (velocity.square().sum(dim=-1) * contact).sum(dim=1)
        * (command_speed(env) > 0.1)
    )


def soft_landing(env):
    landing = first_foot_contact(env)
    impact = torch.linalg.vector_norm(foot_forces(env), dim=-1) * landing
    env.extras.setdefault("log", {})["Metrics/landing_force_mean"] = (
        impact.sum() / landing.sum().clamp_min(1)
    )
    return impact.sum(dim=1) * (command_speed(env) > 0.1)


def stand_still(env):
    return (
        joint_position_error(env).square().sum(dim=1)
        * (command_speed(env) <= 0.1)
    )


def illegal_contact(env):
    sensor = env.scene["contact_forces"]
    ids = indices(env)["contact_other"]
    # Isaac's order is environment, history, body, xyz.
    history = sensor.data.net_forces_w_history[:, :, ids]
    return (
        torch.linalg.vector_norm(history, dim=-1) > 10.0
    ).any(dim=(1, 2))
