"""Register ILIA Reduced velocity environments."""

import gymnasium as gym


for terrain in ("Flat", "Rough"):
    for play in (False, True):
        suffix = "-Play" if play else ""
        env_class = f"IliaReduced{terrain}{'Play' if play else ''}EnvCfg"

        gym.register(
            id=f"Isaac-ILIA-Reduced-Velocity-{terrain}{suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{env_class}",
                "rsl_rl_cfg_entry_point": (
                    f"{__name__}.rsl_rl_ppo_cfg:"
                    f"IliaReduced{terrain}PPORunnerCfg"
                ),
            },
        )
