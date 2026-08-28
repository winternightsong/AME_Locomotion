# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class KuavoS46AME2ActorCriticCfg(RslRlPpoActorCriticCfg):
    """AME terrain encoder with the AME2 global-context branch enabled."""

    attach_global: bool = True

@configclass
class G1AMEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "kuavo_s46_ame_stage1"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticEncoder",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class KuavoS46AME2PPORunnerCfg(G1AMEPPORunnerCfg):
    """Exact policy configuration used by the model-9000 S46 run."""

    max_iterations = 15000
    experiment_name = "kuavo_s46_ame2_stage1_native"
    policy = KuavoS46AME2ActorCriticCfg(
        class_name="ActorCriticEncoder",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        attach_global=True,
    )
