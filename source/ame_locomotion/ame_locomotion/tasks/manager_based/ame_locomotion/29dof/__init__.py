import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

gym.register(
    id="AME-KuavoS46-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "ame_locomotion.tasks.manager_based.ame_locomotion.s46_native."
            "velocity_env_cfg_s46_native:KuavoS46AMEStage1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ame_rsl_rl_ppo_cfg:KuavoS46AME2PPORunnerCfg"
        ),
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AME-KuavoS46-Stage1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "ame_locomotion.tasks.manager_based.ame_locomotion.s46_native."
            "velocity_env_cfg_s46_native:KuavoS46AMEStage1PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ame_rsl_rl_ppo_cfg:KuavoS46AME2PPORunnerCfg"
        ),
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AME-KuavoS46-Stage1-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-KuavoS46-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "ame_locomotion.tasks.manager_based.ame_locomotion.s46_native."
            "velocity_env_cfg_s46_native:KuavoS46AMEStage2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ame_rsl_rl_ppo_cfg:KuavoS46AME2PPORunnerCfg"
        ),
    },
)
