# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class rigid_body_masses(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.sum_mass = torch.sum(
            self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device),
            dim=-1,
        ).unsqueeze(-1)
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.sum_mass = torch.sum(
                self.asset.root_physx_view.get_masses()[:, self.body_ids].to(
                    env.device
                ),
                dim=-1,
            ).unsqueeze(-1)
        return self.sum_mass


class rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(
            None
        ):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        self.idxs = []
        for body_id in self.body_ids:
            idx = sum(self.num_shapes_per_body[:body_id])
            self.idxs.append(idx)

        materials = self.asset.root_physx_view.get_material_properties()
        self.materials = (
            materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
        )

        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            materials = self.asset.root_physx_view.get_material_properties()
            self.materials = (
                materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
            )
        return self.materials


class base_com(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.coms = (
            self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
            .to(env.device)
            .squeeze(1)
        )
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.coms = (
                self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
                .to(env.device)
                .squeeze(1)
            )
        return self.coms


def contact_information(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    data = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    contact_information = torch.sum(torch.square(data), dim=-1) > 1

    return contact_information.float()


def action_delay(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuators_names: str = "base_legs",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.actuators[actuators_names]
        .positions_delay_buffer.time_lags.float()
        .to(env.device)
        .unsqueeze(1)
    )


def joint_torques(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def joint_accs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return contact_force.flatten(1, 2)


def feet_lin_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return body_lin_vel_w.flatten(1)


def push_force(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :]
    return external_force_b.flatten(1)


def push_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_torque_b = asset._external_torque_b[:, asset_cfg.body_ids, :]
    return external_torque_b.flatten(1)


def feet_heights_bipeds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None,
) -> torch.Tensor:
    foot_heights = torch.stack(
        [
            env.scene[sensor_cfg.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [sensor_cfg1, sensor_cfg2]
            if sensor_cfg is not None
        ],
        dim=-1,
    )
    foot_heights = torch.nan_to_num(foot_heights, nan=0, posinf=0, neginf=0)
    foot_heights = torch.clamp(foot_heights - 0.02, min=0.0)

    return foot_heights.flatten(1)


def feet_air_time_obs(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    return air_time


def gait_phase_obs(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
) -> torch.Tensor:
    """Gait phase observation as sin/cos pairs for each leg.

    Outputs a [num_envs, 2*num_legs] tensor. Each leg gets (sin(2π×phase), cos(2π×phase)).
    Parameters ``period`` and ``offset`` must match those used in the ``feet_gait`` reward
    so the policy sees the same phase the reward function evaluates.
    """
    import math

    # episode_length_buf is created after load_managers(), so it doesn't exist
    # when the observation manager calls this function to infer output shape.
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2 * len(offset), device=env.device)

    phase_base = (env.episode_length_buf * env.step_dt) % period / period
    result = []
    for off in offset:
        phase = (phase_base + off) % 1.0
        result.append(torch.sin(2 * math.pi * phase).unsqueeze(1))
        result.append(torch.cos(2 * math.pi * phase).unsqueeze(1))
    return torch.cat(result, dim=-1)


class history_obs(ManagerTermBase):
    """Accumulate history frames of other observation terms in the same group."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        history_length_param = cfg.params.get("history_length", None)
        if history_length_param is None or not isinstance(history_length_param, int) or history_length_param <= 0:
            raise ValueError(
                f"history_obs requires a positive integer 'history_length' parameter, "
                f"but got: {history_length_param}"
            )
        self.history_length: int = history_length_param

        self.obs_term_names: list[str] = []
        self.obs_group_name: str | None = None

        if hasattr(env, "observation_manager"):
            obs_group = None
            for group_name, group in env.observation_manager._groups.items():
                if hasattr(group, "_terms"):
                    for term_name, term in group._terms.items():
                        if term is self:
                            obs_group = group
                            self.obs_group_name = group_name
                            break
                if obs_group is not None:
                    break

            if obs_group is not None and hasattr(obs_group, "_terms"):
                if hasattr(obs_group, "history_length") and obs_group.history_length is not None:
                    raise RuntimeError(
                        "history_obs must be used in ObsGroup with history_length=None. "
                        f"Current group '{self.obs_group_name}' has history_length={obs_group.history_length}"
                    )

                for term_name, term in obs_group._terms.items():
                    if term is not self:
                        self.obs_term_names.append(term_name)

        self.history_buffers: dict[str, torch.Tensor] = {}
        self.obs_dims: dict[str, int] = {}
        self.buffer_initialized = False

    def __call__(self, env: ManagerBasedRLEnv, history_length: int | None = None) -> torch.Tensor:
        """Return concatenated history observations."""
        device = self.device

        if not self.obs_term_names or self.obs_group_name is None:
            return torch.zeros(env.num_envs, 0, device=device, dtype=torch.float32)

        obs_result = env.observation_manager.compute_group(self.obs_group_name)

        if isinstance(obs_result, torch.Tensor):
            obs_group = env.observation_manager._groups.get(self.obs_group_name)
            if obs_group is None or not hasattr(obs_group, "_terms"):
                return torch.zeros(env.num_envs, 0, device=device, dtype=torch.float32)
            obs_dict = {}
            for obs_term_name in self.obs_term_names:
                if obs_term_name in obs_group._terms:
                    obs_term = obs_group._terms[obs_term_name]
                    obs_dict[obs_term_name] = obs_term.compute(env)
        else:
            obs_dict = obs_result

        current_obs_list = []
        valid_names = []

        if not self.buffer_initialized:
            for obs_term_name in self.obs_term_names:
                if obs_term_name not in obs_dict:
                    continue

                current_obs = obs_dict[obs_term_name]
                if current_obs.device != device:
                    current_obs = current_obs.to(device)
                if current_obs.dim() == 1:
                    current_obs = current_obs.unsqueeze(1)

                obs_dim = current_obs.shape[1]
                self.obs_dims[obs_term_name] = obs_dim
                self.history_buffers[obs_term_name] = torch.zeros(
                    env.num_envs,
                    self.history_length,
                    obs_dim,
                    device=device,
                    dtype=torch.float32
                )
                current_obs_list.append((obs_term_name, current_obs))
                valid_names.append(obs_term_name)

            self.buffer_initialized = True
        else:
            for obs_term_name in self.obs_term_names:
                if obs_term_name not in obs_dict:
                    continue

                current_obs = obs_dict[obs_term_name]
                if current_obs.dim() == 1:
                    current_obs = current_obs.unsqueeze(1)
                current_obs_list.append((obs_term_name, current_obs))
                valid_names.append(obs_term_name)

        for obs_term_name, current_obs in current_obs_list:
            history_buffer = self.history_buffers[obs_term_name]

            history_buffer[:, 1:, :] = history_buffer[:, :-1, :]
            history_buffer[:, 0, :] = current_obs

        if valid_names:
            history_list = [
                self.history_buffers[name].view(env.num_envs, -1)
                for name in valid_names
            ]
            return torch.cat(history_list, dim=1)
        else:
            return torch.zeros(env.num_envs, 0, device=device, dtype=torch.float32)
