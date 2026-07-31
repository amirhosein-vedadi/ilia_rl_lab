"""Check an ILIA velocity environment without training."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import ilia_rl_lab.tasks  # noqa: F401
from ilia_rl_lab.assets.robots.ilia_reduced.parameters import (
    ILIA_REDUCED_DOF_NAMES,
)
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=8)
    cfg.seed = 42
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped

    try:
        obs, _ = env.reset()
        rough = base.cfg.scene.terrain.terrain_type == "generator"
        expected = {"actor": 255 if rough else 68, "critic": 270 if rough else 83}

        action = base.action_manager.get_term("joint_pos")
        assert action.joint_names == list(ILIA_REDUCED_DOF_NAMES)
        assert base.action_manager.total_action_dim == 19
        assert abs(base.step_dt - 0.02) < 1.0e-9

        for name, dimension in expected.items():
            assert tuple(obs[name].shape) == (8, dimension), (
                name, obs[name].shape, dimension
            )
            assert torch.isfinite(obs[name]).all()

        print("PASS: observation dimensions, action order, and 50 Hz control.")
        print("Joint order:", action.joint_names)
        print("Joint indices:", action.joint_ids)

        material_event = base.event_manager.get_term_cfg("foot_material").func
        material = base.scene["robot"].root_physx_view.get_material_properties()
        left_start, left_stop = material_event.shape_ranges[0]
        right_start, right_stop = material_event.shape_ranges[1]

        left_mu = material[:, left_start, 0]
        right_mu = material[:, right_start, 0]
        assert torch.allclose(left_mu, right_mu)
        assert ((left_mu >= 0.3) & (left_mu <= 1.6)).all()
        assert torch.allclose(
            material[:, left_start:left_stop, 0],
            material[:, left_start:left_stop, 1],
        )
        assert torch.allclose(
            material[:, right_start:right_stop, 0],
            material[:, right_start:right_stop, 1],
        )
        assert action.encoder_bias.abs().max() <= 0.015001
        print("PASS: shared foot friction and encoder-bias bounds.")

        termination_count = 0
        with torch.inference_mode():
            for step in range(args.steps):
                actions = torch.zeros((8, 19), device=base.device)
                if step >= 20:
                    actions.uniform_(-0.05, 0.05)

                obs, reward, terminated, truncated, _ = env.step(actions)
                termination_count += int((terminated | truncated).sum())

                assert torch.isfinite(reward).all()
                assert torch.isfinite(base.scene["robot"].data.joint_pos).all()
                assert torch.isfinite(base.scene["robot"].data.joint_vel).all()
                assert torch.isfinite(base.scene["robot"].data.root_link_state_w).all()
                for values in obs.values():
                    assert torch.isfinite(values).all()

        # Explicitly exercise a partial reset of the custom state.
        swing = base.reward_manager.get_term_cfg("foot_swing_height").func
        action.previous.fill_(0.2)
        action.previous_previous.fill_(0.1)
        swing.peak.fill_(0.3)
        bias_before = action.encoder_bias.clone()

        ids = torch.tensor([0], device=base.device)
        base._reset_idx(ids)

        assert torch.count_nonzero(action.raw_actions[0]) == 0
        assert torch.count_nonzero(action.previous[0]) == 0
        assert torch.count_nonzero(action.previous_previous[0]) == 0
        assert torch.count_nonzero(swing.peak[0]) == 0
        assert torch.allclose(action.previous[1:], torch.full_like(action.previous[1:], 0.2))
        assert torch.allclose(swing.peak[1:], torch.full_like(swing.peak[1:], 0.3))
        assert torch.equal(action.encoder_bias, bias_before)

        print("PASS: partial reset clears history and swing state only for env 0.")
        print(f"PASS: {args.steps} finite control steps; {termination_count} episode endings.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
