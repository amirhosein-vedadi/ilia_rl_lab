"""Position targets with encoder bias and reset-safe action history."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class OrderedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(self.raw_actions)
        self.previous_previous = torch.zeros_like(self.raw_actions)
        self.encoder_bias = torch.zeros_like(self.raw_actions)

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def joint_names(self):
        return self._joint_names

    def process_actions(self, actions):
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(self.raw_actions)
        super().process_actions(actions)
        self._processed_actions -= self.encoder_bias

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self.previous[ids] = 0.0
        self.previous_previous[ids] = 0.0

        # Startup encoder calibration remains fixed across episode resets.
        self._processed_actions[ids] = self._offset[ids] - self.encoder_bias[ids]


@configclass
class OrderedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = OrderedJointPositionAction
