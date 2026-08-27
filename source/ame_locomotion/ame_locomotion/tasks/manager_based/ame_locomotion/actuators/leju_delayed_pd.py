"""Leju delayed PD actuator, vendored from LejuLab-Train for S46."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class LejuDelayedPDActuator(DelayedPDActuator):
    cfg: "LejuDelayedPDActuatorCfg"

    def __init__(self, cfg: "LejuDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.friction_activation_vel = self._parse_joint_parameter(self.cfg.friction_activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        self.saturation_effort = self._parse_joint_parameter(self.cfg.effort_limit_sim, 0.0)
        self.effort_weaken_velocity_limit = self._parse_joint_parameter(
            self.cfg.effort_weaken_velocity_limit, 0.0
        )

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._joint_vel[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
            - (
                self.friction_static * torch.tanh(joint_vel / self.friction_activation_vel)
                + self.friction_dynamic * joint_vel
            )
        )
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        slope = -self.saturation_effort / (self.velocity_limit - self.velocity_limit / 2)
        effort_limit = torch.clip(
            slope * (self._joint_vel.abs() - self.velocity_limit / 2) + self.saturation_effort,
            min=0.0,
        )
        max_effort = torch.where(
            self._joint_vel.abs() < self.effort_weaken_velocity_limit,
            self.saturation_effort,
            effort_limit,
        )
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, max_effort, self.saturation_effort)
        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class LejuDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = LejuDelayedPDActuator
    effort_weaken_velocity_limit: float | dict[str, float] = 0.0
    friction_static: float | dict[str, float] = 0.0
    friction_activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0.0
