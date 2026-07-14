"""Implicit position drives with mjlab-style per-step command delay."""

from __future__ import annotations

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass


class DelayedImplicitActuator(ImplicitActuator):
    """Delay position targets independently per environment and joint."""

    def __init__(self, cfg, *args, **kwargs):
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError("Require 0 <= min_delay <= max_delay.")
        super().__init__(cfg, *args, **kwargs)
        self._capacity = cfg.max_delay + 1
        self._history = torch.zeros(
            (self._capacity, self._num_envs, self.num_joints),
            device=self._device,
            dtype=self.stiffness.dtype,
        )
        self._lengths = torch.zeros(
            (self._num_envs, 1), device=self._device, dtype=torch.long
        )
        self._env_rows = torch.arange(self._num_envs, device=self._device)[:, None]
        self._joint_cols = torch.arange(self.num_joints, device=self._device)[None, :]
        self._cursor = 0

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._history[:, ids] = 0.0
        self._lengths[ids] = 0
        # Keep the global cursor: other environments may still be running.

    def compute(self, control_action, joint_pos, joint_vel):
        if control_action.joint_positions is None:
            raise ValueError("This actuator requires position targets.")

        self._history[self._cursor].copy_(control_action.joint_positions)
        self._lengths.add_(1).clamp_(max=self._capacity)

        # New independent lag for every joint/environment on every physics step.
        lags = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (self._num_envs, self.num_joints),
            device=self._device,
        )
        # At startup/reset, only use commands from the current episode.
        lags = torch.minimum(lags, self._lengths - 1)
        slots = (self._cursor - lags) % self._capacity
        control_action.joint_positions = self._history[
            slots, self._env_rows, self._joint_cols
        ]
        self._cursor = (self._cursor + 1) % self._capacity

        # PhysX still handles PD control; the parent records torque estimates.
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DelayedImplicitActuator
    min_delay: int = 1
    max_delay: int = 4
