"""Startup randomization, disturbances, and training curricula."""

import torch

from isaaclab.managers import ManagerTermBase

from .terms import action_term, command, robot


def randomize_encoder_bias(env, env_ids, bias_range=(-0.015, 0.015)):
    term = action_term(env)
    ids = slice(None) if env_ids is None else env_ids
    term.encoder_bias[ids] = torch.empty_like(
        term.encoder_bias[ids]
    ).uniform_(*bias_range)


class SharedFootMaterial(ManagerTermBase):
    """One friction sample per environment, shared by both feet."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = robot(env)
        foot_ids, _ = self.asset.find_bodies(
            env.cfg.foot_body_names, preserve_order=True
        )

        # Same shape-index discovery used by Isaac Lab's material event.
        counts = []
        for path in self.asset.root_physx_view.link_paths[0]:
            view = self.asset._physics_sim_view.create_rigid_body_view(path)
            counts.append(view.max_shapes)

        assert sum(counts) == self.asset.root_physx_view.max_shapes
        self.shape_ranges = [
            (sum(counts[:i]), sum(counts[:i + 1])) for i in foot_ids
        ]
        assert all(stop > start for start, stop in self.shape_ranges)

    def __call__(self, env, env_ids, mu_range=(0.3, 1.6), num_buckets=64):
        ids = (
            torch.arange(env.num_envs, device="cpu")
            if env_ids is None else env_ids.cpu()
        )

        # Buckets limit the number of distinct PhysX materials.
        buckets = torch.empty(num_buckets, device="cpu").uniform_(*mu_range)
        mu = buckets[torch.randint(num_buckets, (len(ids),), device="cpu")]

        materials = self.asset.root_physx_view.get_material_properties().clone()
        # Columns: static friction, dynamic friction, restitution.
        materials[ids] = 0.0

        for start, stop in self.shape_ranges:
            materials[ids, start:stop, :2] = mu[:, None, None]

        self.asset.root_physx_view.set_material_properties(materials, ids)


def push_root_velocity(env, env_ids, velocity_range):
    asset = robot(env)
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None else env_ids
    )
    ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0))
         for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=env.device,
    )
    delta = torch.rand((len(ids), 6), device=env.device)
    delta = ranges[:, 0] + delta * (ranges[:, 1] - ranges[:, 0])
    velocity = asset.data.root_link_vel_w[ids].clone() + delta
    asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)


def command_curriculum(env, env_ids):
    del env_ids
    ranges = env.command_manager.get_term("twist").cfg.ranges
    stage_number = 0
    for stage_number_candidate, stage in enumerate(env.cfg.velocity_stages, 1):
        if env.common_step_counter > stage["step"]:
            ranges.lin_vel_x = stage["lin_vel_x"]
            ranges.lin_vel_y = stage["lin_vel_y"]
            ranges.ang_vel_z = stage["ang_vel_z"]
            stage_number = stage_number_candidate
    return float(stage_number)


def terrain_curriculum(env, env_ids):
    terrain = env.scene.terrain

    # Do not advance terrain during an initial/manual reset with no episode.
    ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(ids) > 0:
        distance = torch.linalg.vector_norm(
            robot(env).data.root_link_pos_w[ids, :2]
            - env.scene.env_origins[ids, :2],
            dim=1,
        )
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2.0
        requested_distance = (
            torch.linalg.vector_norm(command(env)[ids, :2], dim=1)
            * env.cfg.episode_length_s
        )
        move_down = (distance < 0.5 * requested_distance) & ~move_up
        terrain.update_env_origins(ids, move_up, move_down)

    return terrain.terrain_levels.float().mean()


def randomize_terrain(env, env_ids):
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return

    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None else env_ids
    )
    rows, cols = terrain.terrain_origins.shape[:2]
    terrain.terrain_levels[ids] = torch.randint(
        rows, (len(ids),), device=env.device
    )
    terrain.terrain_types[ids] = torch.randint(
        cols, (len(ids),), device=env.device
    )
    terrain.env_origins[ids] = terrain.terrain_origins[
        terrain.terrain_levels[ids], terrain.terrain_types[ids]
    ]
