from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.managers.manager_term_cfg import RewardTermCfg


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_clip(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, min=0.0, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power (torque * velocity) on the articulation using L1 norm.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying the robot and joints. If None, defaults to "robot".
        joint_names: Joint names to specify joints directly. Can be a list of strings or a single string (regex pattern).
                    If provided, will override joint_names in asset_cfg. If both are None, uses all joints.

    Returns:
        Sum of absolute joint power (torque * velocity) for the specified joints.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)


class action_smoothness_l2(ManagerTermBase):
    """Penalize the second-order rate of change of actions using L2 squared kernel.

    This reward term penalizes action jerk (second derivative of actions) to encourage
    smooth control. It requires maintaining the previous-previous action state, which
    is not provided by Isaac Lab's action_manager (only prev_action is available).

    Therefore, this must be implemented as a class to maintain state across steps.
    """

    def __init__(self, cfg: "RewardTermCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = None

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Compute action smoothness penalty (second-order difference).

        Args:
            env: The environment instance.

        Returns:
            Sum of squared second-order action differences.
        """
        # Initialize on first call
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()

        # Compute second-order difference: action - 2*prev_action + prev_prev_action
        action_smoothness_l2 = torch.sum(
            torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            ),
            dim=1,
        )

        # Update state for next step
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    return torch.square(base_height - target_height)


def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L1 kernel.

    Compared to L2 squared, L1 provides constant gradient regardless of
    deviation magnitude, giving consistent push toward target height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )
    return torch.abs(base_height - target_height)


# def track_lin_vel_xy_yaw_frame_exp(
#     env,
#     std: float,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_rotate_inverse(
#         yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3]
#     )
#     lin_vel_error = torch.sum(
#         torch.square(
#             env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
#         ),
#         dim=1,
#     )
#     return torch.exp(-lin_vel_error / std**2)


# def track_ang_vel_z_world_exp(
#     env,
#     command_name: str,
#     std: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(
#         env.command_manager.get_command(command_name)[:, 2]
#         - asset.data.root_com_ang_vel_w[:, 2]
#     )
#     return torch.exp(-ang_vel_error / std**2)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, violation_max: float = torch.inf) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0, max=violation_max), dim=1)


def contact_forces_on_landing(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    violation_max: float = torch.inf,
) -> torch.Tensor:
    """Penalize landing impact peak only at first_contact moment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 检测 air→ground 转换（每脚每次落地恰好触发一次）
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # 在该 control step 的 history 内取垂直分量峰值
    # 仅 z 分量，避免 policy 通过倾斜脚把垂直冲击转成水平动量的 gaming
    net_force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak_force = torch.max(torch.abs(net_force_history[..., 2]), dim=1)[0]

    violation = (peak_force - threshold).clip(min=0.0, max=violation_max)

    # 仅 first_contact 帧记账，其他帧 reward = 0
    return torch.sum(violation * first_contact.float(), dim=1)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on joint deviation from default when commanded to stand still."""
    asset: Articulation = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos_nominal
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff_angle = current_joint_pos - default_joint_pos
    reward = torch.sum(torch.abs(diff_angle), dim=-1)

    command = env.command_manager.get_command(command_name)
    cmd_low = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) < 0.05

    reward *= cmd_low.float()
    return reward


def joint_deviation_pos_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_dev: float = 0.6,
    std: float = 0.3,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - joint_default), dim=-1)

    error = torch.square(deviation - target_dev)
    reward = torch.exp(-error / (std ** 2))

    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= (cmd_norm > command_threshold).float()
    return reward


def hip_pitch_phase_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float = 0.7,
    amplitude: float = 0.4,
    std: float = 0.3,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Phase-locked sinusoidal hip pitch swing reward during walking.

    Unlike joint_deviation_pos_reward (which is gameable by static large angles),
    this reward requires actual dynamic oscillation matching a clock-based sin target.
    Left and right legs are in anti-phase (offset 0.5).

    Args:
        asset_cfg: must resolve to exactly 2 hip pitch joints in alphabetical order
                   (idx 0 = left, idx 1 = right). Use joint_names=["leg_[l,r]4_joint"].
        period: gait period in seconds (~0.7s for natural humanoid walking).
        amplitude: sin amplitude in rad (~0.4 = 23° hip pitch swing).
        std: gaussian width — std=0.3 gives reward 0.8 at error 0.02, 0.14 at error 0.18.
        command_threshold: reward = 0 when linear cmd norm < threshold.

    Math verification:
        - Perfect dynamic swing (both legs follow sin): reward ≈ 1.0 always.
        - Optimal static (0, 0): avg reward ≈ exp(-0.16/0.09) ≈ 0.17.
        - Asymmetric static (+0.3, -0.3): avg reward ≈ 0.13.
        → PPO can NOT cheat by static pose, must dynamically oscillate.
    """
    import math
    asset: Articulation = env.scene[asset_cfg.name]

    # Clock-based phase, [num_envs]
    phase = (env.episode_length_buf.float() * env.step_dt) % period / period

    # Anti-phase sin targets, [num_envs]
    left_target = amplitude * torch.sin(2 * math.pi * phase)
    right_target = amplitude * torch.sin(2 * math.pi * (phase + 0.5))

    # Joint deviation from default, [num_envs, 2]
    leg_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    leg_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dev = leg_pos - leg_default

    # Squared error sum (left + right), exp kernel
    error = (dev[:, 0] - left_target) ** 2 + (dev[:, 1] - right_target) ** 2
    reward = torch.exp(-error / (std ** 2))

    # cmd mask (linear xy only, matches feet_air_time / alternating_contacts)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= (cmd_norm > command_threshold).float()
    return reward


def hip_pitch_stance_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    amplitude: float = 0.3,
    std: float = 0.3,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Contact-synced hip pitch swing reward .

    Unlike clock-based phase (D13: PPO has no episode_length observation, can't learn),
    this reward uses CURRENT foot contact state to define hip pitch target:
        - Stance leg (foot in contact): target = -amplitude (backward, propulsion)
        - Swing leg (foot in air):       target = +amplitude (forward, recovery)

    PPO can infer contact state from joint pos/vel observations, so this reward
    is learnable by a feedforward policy. Auto-syncs to PPO's natural step period.

    Args:
        sensor_cfg: contact sensor on both feet (alphabetical: left=0, right=1).
        asset_cfg: hip pitch joints (alphabetical: leg_l4=0, leg_r4=1).
        amplitude: hip pitch swing magnitude in rad (~0.3 = 17°).
        std: gaussian width of reward kernel.
        command_threshold: linear cmd norm gate; reward = 0 below this.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # [N, 2]

    # Stance → -A, Swing → +A
    target_l = torch.where(is_contact[:, 0], -amplitude, amplitude)
    target_r = torch.where(is_contact[:, 1], -amplitude, amplitude)

    asset: Articulation = env.scene[asset_cfg.name]
    leg_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    leg_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dev = leg_pos - leg_default

    error = (dev[:, 0] - target_l) ** 2 + (dev[:, 1] - target_r) ** 2
    reward = torch.exp(-error / (std ** 2))

    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= (cmd_norm > command_threshold).float()
    return reward


def joint_vel_at_rest(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on joint velocity when commanded to stand still."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    cmd_low = (
        torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    ) < 0.05

    return joint_vel_sq * cmd_low.float()


def flat_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on torso tilt (projected gravity horizontal components).

    R25: complements flat_orientation_l2 which uses L2 kernel and is insensitive
    to small tilt angles (<5°). L1 provides stronger gradient at small angles,
    preventing the policy from settling into shallow forward lean at cmd=0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1)


def feet_on_ground_at_rest(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward both feet on ground when commanded to stand still.

    R24: strong positive reward for cmd=0 standing. Returns 1.0 when both feet
    are in contact AND command is near zero, 0 otherwise. Used with weight +3.0
    to strongly encourage the "stand at attention" behavior at zero command.

    Does NOT affect walking (cmd > 0.05) since zero_flag masks it out.
    Does NOT constrain joint motion - policy is free to adjust joints for
    balance, as long as both feet stay on the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    both_on_ground = (is_contact.int().sum(dim=1) == 2).float()

    command = env.command_manager.get_command(command_name)
    zero_flag = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) < 0.05
    return both_on_ground * zero_flag.float()


def stand_still_base_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on base velocity when commanded to stand still.

    R22: directly penalizes any base motion at zero command. Complements
    stand_still_without_cmd (joint level) to robustly enforce zero-velocity
    standing. Uses L2 so small perturbation drift is barely penalized but
    sustained drift (shuffling) gets strongly penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)
    ang_sq = torch.square(asset.data.root_ang_vel_w[:, 2])
    reward = vel_sq + 0.5 * ang_sq

    command = env.command_manager.get_command(command_name)
    zero_flag = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) < 0.05
    return reward * zero_flag.float()


def yaw_drift_when_straight(
    env: ManagerBasedRLEnv,
    command_name: str,
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on actual yaw velocity when commanded to go straight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_ang_z = env.command_manager.get_command(command_name)[:, 2]
    actual_yaw_vel = asset.data.root_ang_vel_w[:, 2]

    # 仅 cmd_ang_z 接近 0 时（直行场景）激活
    straight_flag = (torch.abs(cmd_ang_z) < cmd_threshold).float()
    return torch.square(actual_yaw_vel) * straight_flag


def lateral_joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_pairs: list,
) -> torch.Tensor:
    """Penalize anti-symmetric deviation in lateral joints (creates torsion).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_lateral_mirror_cache") or env._lateral_mirror_cache is None:
        env._lateral_mirror_cache = []
        for left_name, right_name in joint_pairs:
            left_id = asset.find_joints(left_name)[0][0]
            right_id = asset.find_joints(right_name)[0][0]
            env._lateral_mirror_cache.append((left_id, right_id))

    reward = torch.zeros(env.num_envs, device=env.device)
    for left_id, right_id in env._lateral_mirror_cache:
        diff = asset.data.joint_pos[:, left_id] - asset.data.joint_pos[:, right_id]
        reward += torch.square(diff)
    return reward / max(len(env._lateral_mirror_cache), 1)


# def joint_deviation_waist_l1(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize waist joint deviation from default position using L1 norm."""
#     asset: Articulation = env.scene[asset_cfg.name]
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def arm_swing_coordination(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    amplitude: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward contralateral arm-leg coordination.

    Left arm follows right leg phase, right arm follows left leg phase.
    Uses exponential kernel to reward matching a sinusoidal swing target.
    """
    import math

    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    shoulder_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = shoulder_pos - shoulder_default  # [num_envs, 2]: left, right

    phase_base = (env.episode_length_buf * env.step_dt) % period / period

    # Contralateral: left arm (idx 0) follows right leg (offset[1]),
    #                right arm (idx 1) follows left leg (offset[0])
    left_arm_target = amplitude * torch.sin(2 * math.pi * ((phase_base + offset[1]) % 1.0))
    right_arm_target = amplitude * torch.sin(2 * math.pi * ((phase_base + offset[0]) % 1.0))

    error = (deviation[:, 0] - left_arm_target) ** 2 + (deviation[:, 1] - right_arm_target) ** 2
    reward = torch.exp(-error / (std**2))

    return reward


def alternating_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_stance_time: float = 0.4,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward single-leg stance with time decay + command-based masking.

    R16: adds max_stance_time (time decay to prevent hopping).
    R20: adds command mask (cmd < threshold → reward=0, follows ETH legged_gym).
    When combined with stand_still_without_cmd, enables zero-velocity standing.

    Args:
        max_stance_time: stance time (s) after which reward starts decaying.
        command_name: if provided, mask reward at zero command.
        command_threshold: minimum command norm to enable stepping reward (m/s).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    n_contacts = is_contact.int().sum(dim=1)

    single = (n_contacts == 1).float()

    # Time decay: discourage staying on one foot too long
    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    max_contact = contact_times.max(dim=1)[0]
    time_factor = torch.clamp(
        1.0 - (max_contact - max_stance_time) / max_stance_time, 0.0, 1.0
    )

    reward = single * time_factor - 0.1 * (n_contacts == 0).float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        reward = reward * (cmd_norm > command_threshold).float()

    return reward


def contralateral_arm_swing(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg,
    leg_cfg: SceneEntityCfg,
    scale: float = 1.0,
    std: float = 0.15,
) -> torch.Tensor:
    """Position-based contralateral arm-leg coordination, bounded [0, 1].

    Strategy: when right leg is more active (knee bent more, hip forward),
    left arm swings forward, right arm swings backward. When legs are equal
    (standing, crouching), targets are 0 → arms at default.

    Args:
        scale: arm amplitude (typically 1.0 for 1:1 following).
        std: tolerance for position matching (smaller = stricter).
    """
    asset: Articulation = env.scene[arm_cfg.name]
    arm_dev = (asset.data.joint_pos[:, arm_cfg.joint_ids]
               - asset.data.default_joint_pos[:, arm_cfg.joint_ids])
    leg_dev = (asset.data.joint_pos[:, leg_cfg.joint_ids]
               - asset.data.default_joint_pos[:, leg_cfg.joint_ids])

    #zero-mean signal (right - left) to avoid positional bias
    diff = leg_dev[:, 1] - leg_dev[:, 0]

    # Contralateral: right leg active → left arm forward (-), right arm backward (+)
    target_left_arm = -scale * diff
    target_right_arm = +scale * diff

    error = (arm_dev[:, 0] - target_left_arm) ** 2 + (arm_dev[:, 1] - target_right_arm) ** 2
    return torch.exp(-error / (std ** 2))


def feet_landing_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """[DEPRECATED in R14] Penalize downward foot velocity at first contact.

    Timing issue: foot_vel_z at first_contact is post-impact velocity (already
    decelerated by ground reaction). Use feet_descent_velocity instead, which
    measures velocity during the descent phase before contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    landing_speed = torch.clamp(-foot_vel_z - threshold, min=0.0) * first_contact
    return torch.sum(landing_speed, dim=1)


def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward foot clearance during swing phase.

    Uses foot link world z position. On rough terrain (small ±5cm variation),
    a wider std absorbs the terrain noise. Avoids dependency on scanner sensors.

    Active when foot is in air (not contacting ground). Uses exponential
    kernel on the height matching, bounded [0, 1] per foot.

    Args:
        sensor_cfg: contact sensor for both feet (used to detect swing phase).
        asset_cfg: robot asset with both feet body_ids in alphabetical order [l, r].
        target_height: desired foot link world z position when in swing (m).
        std: standard deviation of exponential kernel.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    in_swing = ~is_contact

    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [num_envs, 2] world z

    error = torch.square(foot_z - target_height)
    reward = torch.exp(-error / (std ** 2)) * in_swing.float()
    return torch.sum(reward, dim=1)


def feet_descent_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_threshold: float = 0.10,
) -> torch.Tensor:
    """Penalize foot descent velocity when approaching the ground.

    This version provides dense, proactive feedback during the descent phase,
    teaching the policy to slow down BEFORE impact.

    Active when:
    - Foot is in air (no current contact)
    - Foot ankle height in world frame < height_threshold (close to ground)
    - Foot is descending (z velocity < 0)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    in_air = ~is_contact
    close_to_ground = foot_pos_z < height_threshold
    descending = foot_vel_z < 0

    active = (in_air & close_to_ground & descending).float()
    descent_speed = torch.abs(foot_vel_z) * active
    return torch.sum(descent_speed, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward for lifting feet to target height during swing phase."""
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def body_lin_acc_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies in z-axis direction using L2-kernel.

    Args:
        env: The environment instance.
        asset_cfg: Asset configuration specifying the robot and bodies.

    Returns:
        Sum of squared z-axis linear accelerations for all specified bodies.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2]), dim=1)


def feet_heading_alignment_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for aligning foot heading with command heading using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term(command_name)
    heading_angle = getattr(command_term, "heading_target")

    command_direction_xy = torch.stack([
        torch.cos(heading_angle),
        torch.sin(heading_angle)
    ], dim=1)

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs = foot_quat_w.shape[0]
    num_feet = foot_quat_w.shape[1]

    ang_vel_command = env.command_manager.get_command(command_name)[:, 2]

    heading_error = torch.abs(heading_angle - asset.data.heading_w)
    heading_error = torch.minimum(heading_error, 2 * torch.pi - heading_error)
    is_aligned = heading_error < 0.1

    is_turning_left = (ang_vel_command > 0.01) & (~is_aligned)
    is_turning_right = (ang_vel_command < -0.01) & (~is_aligned)

    is_left_foot = torch.zeros(num_feet, dtype=torch.bool, device=env.device)
    is_left_foot[0] = True

    foot_local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(num_envs * num_feet, 3)

    foot_quat_flat = foot_quat_w.reshape(-1, 4)
    foot_x_world = quat_rotate(foot_quat_flat, foot_local_x)
    foot_x_world = foot_x_world.reshape(num_envs, num_feet, 3)

    foot_x_xy = foot_x_world[:, :, :2]
    foot_x_xy_norm = torch.norm(foot_x_xy, dim=2, keepdim=True)
    foot_x_xy_norm = torch.clamp(foot_x_xy_norm, min=1e-6)
    foot_x_xy_normalized = foot_x_xy / foot_x_xy_norm

    command_direction_xy_expanded = command_direction_xy.unsqueeze(1).expand(-1, num_feet, -1)

    alignment = torch.sum(foot_x_xy_normalized * command_direction_xy_expanded, dim=2)

    alignment_adjusted = alignment.clone()

    is_turning_left_expanded = is_turning_left.unsqueeze(1).expand(-1, num_feet)
    is_turning_right_expanded = is_turning_right.unsqueeze(1).expand(-1, num_feet)
    is_left_foot_expanded = is_left_foot.unsqueeze(0).expand(num_envs, -1)

    left_turn_mask = is_turning_left_expanded & (~is_left_foot_expanded)
    alignment_adjusted[left_turn_mask] = -alignment[left_turn_mask]

    right_turn_mask = is_turning_right_expanded & is_left_foot_expanded
    alignment_adjusted[right_turn_mask] = -alignment[right_turn_mask]

    alignment_error = 1.0 - alignment_adjusted

    reward_per_foot = torch.exp(-alignment_error / (std ** 2))

    reward = torch.mean(reward_per_foot, dim=1)

    return reward


def body_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 0.2,
    max_dist: float = 0.5,
) -> torch.Tensor:
    """Reward based on distance between two bodies. Penalize bodies too close or too far."""
    asset: Articulation = env.scene[asset_cfg.name]

    if len(asset_cfg.body_ids) != 2:
        raise ValueError(f"body_distance requires exactly 2 bodies, but got {len(asset_cfg.body_ids)}")

    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    body_dist = torch.norm(body_pos[:, 0, :] - body_pos[:, 1, :], dim=1)

    d_min = torch.clamp(body_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(body_dist - max_dist, 0.0, 0.5)

    reward = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2.0

    return reward

def joint_deviation_l1_no_yaw_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation only when yaw command is absent or near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(torch.abs(diff_angle), dim=-1)
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] > 2:
        mask = torch.abs(cmd[:, 2]) < yaw_threshold
    else:
        mask = torch.ones(cmd.shape[0], device=reward.device, dtype=torch.bool)
    reward *= mask
    return reward


def joint_power_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)


def gravity_aligned_when_stopping(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upright torso pitch when the velocity command is near zero.

    Migrated from leju_robot_rl S42 task. Only when ||cmd_xy|| < 0.05 (i.e. "stop"
    command), reward exp(-5 * pitch^2) so that the robot stays vertically aligned.
    """
    is_zero_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.05
    asset: Articulation = env.scene[asset_cfg.name]

    root_quat = asset.data.root_link_quat_w
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    pitch = torch.asin(2.0 * (w * y - x * z))

    reward = torch.exp(-5.0 * torch.square(pitch))
    masked_reward = torch.zeros_like(reward)
    masked_reward[is_zero_cmd] = reward[is_zero_cmd]
    return masked_reward
