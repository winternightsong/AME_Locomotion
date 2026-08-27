#!/usr/bin/env python3
"""AME Stage 2 Readiness Evaluator

Evaluates a trained policy against Stage 2 transition criteria by running
the policy on controlled terrain levels and types, collecting per-episode
statistics.

Usage on 七号机:
    cd /data/song/projects/AME_Locomotion
    python scripts/rsl_rl/stage2_eval.py \
        --checkpoint /path/to/model_XXXX.pt \
        --task AME-S46-Stage1-Play-v0 \
        --num_envs 128 \
        --episodes 30 \
        --gpu 1
"""

import argparse
import sys
import os
import time
import json
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description="AME Stage 2 Readiness Evaluator")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--task", type=str, default="AME-S46-Stage1-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=30, help="Episodes per environment")
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--output", type=str, default="stage2_eval_results.json")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks
import ame_locomotion.tasks

TERRAIN_TYPE_NAMES = [
    "pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
    "hf_pyramid_slope", "hf_pyramid_slope_inv", "deep_pits",
    "narrow_bridge", "plum_blossom_stakes",
]

def main():
    device = f"cuda:{args_cli.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args_cli.checkpoint}")

    env = ManagerBasedRLEnv(args_cli.task, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env)

    # Load the inference policy directly from the environment
    # The env's agent has the policy already loaded via the task config
    unwrapped = env.unwrapped
    agent = unwrapped.agent

    # Get the policy
    try:
        policy = agent.alg.policy
    except AttributeError:
        try:
            policy = agent.alg.actor_critic
        except AttributeError:
            policy = None

    if policy is None:
        print("ERROR: Could not get policy from agent. Exiting.")
        return

    policy.to(device)
    policy.eval()

    # Load checkpoint weights
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load into policy
    try:
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as e:
        print(f"Warning: Could not load checkpoint weights: {e}")

    num_episodes_per_env = args_cli.episodes
    num_envs = args_cli.num_envs

    # Accumulators
    level_success = defaultdict(list)
    type_success = defaultdict(list)
    all_survival_times = []
    all_vel_errors_xy = []
    all_vel_errors_yaw = []
    all_joint_limit_flags = []
    all_torque_values = []
    all_foot_slide_flags = []

    total_steps = 0
    total_episodes = 0
    total_successes = 0

    max_ep_steps = int(unwrapped.max_episode_length)

    terrain = unwrapped.scene.terrain
    terrain_levels = terrain.terrain_levels.cpu().clone()
    terrain_types = terrain.terrain_types.cpu().clone()

    print(f"Max episode steps: {max_ep_steps}")
    print(f"Initial terrain levels: {sorted(terrain_levels.unique().tolist())}")
    print(f"Initial terrain types: {sorted(terrain_types.unique().tolist())}")

    env_episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    env_finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_terrain_level = terrain_levels.clone()
    episode_terrain_type = terrain_types.clone()

    env.reset()
    start_time = time.time()

    while not env_finished.all():
        obs = env.get_observations()

        with torch.no_grad():
            actions = policy.act(obs)
            if isinstance(actions, tuple):
                actions = actions[0]
            actions = actions.clamp(-1.0, 1.0)

        obs, rewards, dones, infos = env.step(actions)
        total_steps += num_envs

        # Get state data
        robot = unwrapped.scene["robot"]
        cmd_term = unwrapped.command_manager.get_term("base_velocity")

        # Velocity error
        actual_lin_vel_b = robot.data.root_lin_vel_b[:, :2]
        actual_ang_vel_b = robot.data.root_ang_vel_b[:, 2]
        cmd_vel_b = cmd_term.command

        vel_err_xy = torch.norm(cmd_vel_b[:, :2] - actual_lin_vel_b, dim=1)
        vel_err_yaw = torch.abs(cmd_vel_b[:, 2] - actual_ang_vel_b)

        # Joint proximity to soft limits
        joint_pos = robot.data.joint_pos
        soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
        soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]
        joint_range = soft_upper - soft_lower + 1e-8

        prox_lower = (joint_pos - soft_lower) / joint_range
        prox_upper = (soft_upper - joint_pos) / joint_range
        min_prox = torch.min(prox_lower, prox_upper)
        any_joint_at_limit = (min_prox < 0.02).any(dim=1)

        # Torques
        torques = robot.data.applied_torque
        all_torque_values.extend(torques.abs().cpu().tolist())

        # Foot sliding: foot in contact while moving
        contact_forces = unwrapped.scene["contact_forces"]
        foot_contact_mask = (contact_forces.data.force_matrix_w > 1.0).any(dim=-1)
        # Approximate: if foot contact force > threshold and horizontal foot vel > threshold
        any_foot_sliding = torch.zeros(num_envs, dtype=torch.bool, device=device)
        try:
            foot_bodies = robot.find_bodies("leg_l6_link,leg_r6_link")
            if foot_bodies:
                foot_vel_w = robot.data.body_lin_vel_w[:, foot_bodies, :2]
                foot_speed = foot_vel_w.norm(dim=-1)
                sliding = (foot_speed > 0.1) & foot_contact_mask[:, foot_bodies].any(dim=-1)
                any_foot_sliding = sliding.any(dim=1)
        except Exception:
            pass

        # Accumulate for active envs
        active_mask = ~env_finished
        all_vel_errors_xy.extend(vel_err_xy[active_mask].cpu().tolist())
        all_vel_errors_yaw.extend(vel_err_yaw[active_mask].cpu().tolist())
        all_joint_limit_flags.extend(any_joint_at_limit[active_mask].cpu().tolist())
        all_foot_slide_flags.extend(any_foot_sliding[active_mask].cpu().tolist())

        episode_steps[active_mask] += 1

        newly_done = dones & active_mask
        if newly_done.any():
            done_idx = newly_done.nonzero(as_tuple=True)[0]

            for idx in done_idx:
                level = episode_terrain_level[idx].item()
                ttype = episode_terrain_type[idx].item()
                steps = episode_steps[idx].item()
                success = (steps >= max_ep_steps - 1)

                level_success[level].append(success)
                type_success[ttype].append(success)
                all_survival_times.append(steps)
                total_episodes += 1
                if success:
                    total_successes += 1

                episode_steps[idx] = 0
                env_episode_count[idx] += 1

                if env_episode_count[idx] >= num_episodes_per_env:
                    env_finished[idx] = True

            env.reset(env_ids=newly_done)
            new_levels = terrain.terrain_levels.cpu()
            new_types = terrain.terrain_types.cpu()
            episode_terrain_level[newly_done] = new_levels[newly_done]
            episode_terrain_type[newly_done] = new_types[newly_done]

        if total_episodes > 0 and total_episodes % 20 == 0:
            elapsed = time.time() - start_time
            rate = total_episodes / elapsed if elapsed > 0 else 0
            pct = 100 * total_episodes / (num_envs * num_episodes_per_env)
            succ_rate = 100 * total_successes / max(1, total_episodes)
            print(f"  [{pct:.0f}%] Episodes: {total_episodes}, "
                  f"Success: {total_successes}/{total_episodes} ({succ_rate:.1f}%), "
                  f"Rate: {rate:.1f} eps/s, Steps: {total_steps}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete: {elapsed:.1f}s ({total_episodes} episodes, {total_steps} steps)")

    # ========== Compute Results ==========
    print("\n" + "=" * 80)
    print("STAGE 2 READINESS EVALUATION RESULTS")
    print("=" * 80)

    results = {}

    overall_success = total_successes / max(1, total_episodes)
    results["overall_success_rate"] = overall_success
    print(f"\n1. Overall Success Rate: {overall_success*100:.1f}% "
          f"(target >=85%, recommended >=90%)")

    level_0_6 = []
    level_7_9 = []
    print(f"\n2. Per-Level Success Rates:")
    for level in range(10):
        if level in level_success:
            succ = sum(level_success[level])
            n = len(level_success[level])
            rate = succ / n
            print(f"   Level {level}: {rate*100:.1f}% (n={n}, success={succ})")
            if level <= 6:
                level_0_6.extend(level_success[level])
            else:
                level_7_9.extend(level_success[level])

    rate_0_6 = sum(level_0_6) / len(level_0_6) if level_0_6 else None
    rate_7_9 = sum(level_7_9) / len(level_7_9) if level_7_9 else None
    if rate_0_6 is not None:
        results["level_0_6_success_rate"] = rate_0_6
        print(f"   Levels 0-6: {rate_0_6*100:.1f}% (target >=95%)")
    if rate_7_9 is not None:
        results["level_7_9_success_rate"] = rate_7_9
        print(f"   Levels 7-9: {rate_7_9*100:.1f}% (target >=70%)")

    print(f"\n3. Per-Terrain-Type Success Rates:")
    for ttype in range(len(TERRAIN_TYPE_NAMES)):
        name = TERRAIN_TYPE_NAMES[ttype]
        if ttype in type_success:
            all_rates = type_success[ttype]
            rate = sum(all_rates) / len(all_rates)
            n = len(all_rates)
            results["type_" + name + "_success_rate"] = rate
            print(f"   {name}: {rate*100:.1f}% (n={n})")

    if all_survival_times:
        survival_20s = sum(1 for s in all_survival_times if s >= max_ep_steps - 1) / len(all_survival_times)
        results["survival_20s_rate"] = survival_20s
        print(f"\n4. 20s Episode Survival: {survival_20s*100:.1f}% (target >=85%)")

    illegal_contact = 1.0 - overall_success
    results["illegal_contact_rate"] = illegal_contact
    print(f"\n5. Illegal Contact Termination: {illegal_contact*100:.1f}% (target <10%, recommended <5%)")

    if all_vel_errors_xy:
        arr = np.array(all_vel_errors_xy)
        rmse_vel_xy = float(np.sqrt(np.mean(np.square(arr))))
        mae_vel_xy = float(np.mean(arr))
        p95_vel_xy = float(np.percentile(arr, 95))
        results["rmse_linear_vel"] = rmse_vel_xy
        results["mae_linear_vel"] = mae_vel_xy
        results["p95_linear_vel"] = p95_vel_xy
        print(f"\n6. Linear Velocity Error: RMSE={rmse_vel_xy:.4f}, MAE={mae_vel_xy:.4f}, P95={p95_vel_xy:.4f} m/s")
        print(f"   (target RMSE <0.30, recommended <0.20-0.25)")

    if all_vel_errors_yaw:
        arr = np.array(all_vel_errors_yaw)
        rmse_vel_yaw = float(np.sqrt(np.mean(np.square(arr))))
        mae_vel_yaw = float(np.mean(arr))
        p95_vel_yaw = float(np.percentile(arr, 95))
        results["rmse_yaw_rate"] = rmse_vel_yaw
        results["mae_yaw_rate"] = mae_vel_yaw
        results["p95_yaw_rate"] = p95_vel_yaw
        print(f"\n7. Yaw Rate Error: RMSE={rmse_vel_yaw:.4f}, MAE={mae_vel_yaw:.4f}, P95={p95_vel_yaw:.4f} rad/s")
        print(f"   (target RMSE <0.35, recommended <0.25-0.30)")

    if all_joint_limit_flags:
        jl_pct = 100.0 * np.mean([float(x) for x in all_joint_limit_flags])
        results["joint_limit_timestep_pct"] = jl_pct
        print(f"\n8. Joint Limit Triggers: {jl_pct:.2f}% of timesteps (target <2%)")

    if all_torque_values:
        arr = np.array(all_torque_values)
        results["max_torque"] = float(arr.max())
        results["mean_torque"] = float(arr.mean())
        results["p99_torque"] = float(np.percentile(arr, 99))
        print(f"\n9. Torque: Max={arr.max():.2f}, Mean={arr.mean():.2f}, P99={np.percentile(arr, 99):.2f} Nm")

    if all_foot_slide_flags:
        fs_pct = 100.0 * np.mean([float(x) for x in all_foot_slide_flags])
        results["foot_slide_timestep_pct"] = fs_pct
        print(f"\n10. Foot Sliding: {fs_pct:.2f}% of timesteps (target <10% episodes)")

    if all_survival_times:
        arr = np.array(all_survival_times)
        results["episode_length_mean"] = float(arr.mean())
        results["episode_length_min"] = float(arr.min())
        results["episode_length_p5"] = float(np.percentile(arr, 5))
        print(f"\n11. Episode Lengths: Mean={arr.mean():.1f}, Min={arr.min()}, P5={np.percentile(arr, 5):.1f} steps")

    results["total_episodes"] = total_episodes
    results["total_steps"] = total_steps
    results["elapsed_seconds"] = elapsed

    with open(args_cli.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args_cli.output}")

    # Verdict
    print("\n" + "=" * 80)
    print("STAGE 2 READINESS VERDICT")
    print("=" * 80)

    checks = []
    checks.append(("Overall success >= 85%", overall_success >= 0.85, overall_success))
    if rate_0_6 is not None:
        checks.append(("Level 0-6 success >= 95%", rate_0_6 >= 0.95, rate_0_6))
    if rate_7_9 is not None:
        checks.append(("Level 7-9 success >= 70%", rate_7_9 >= 0.70, rate_7_9))
    if "survival_20s_rate" in results:
        checks.append(("20s survival >= 85%", results["survival_20s_rate"] >= 0.85, results["survival_20s_rate"]))
    checks.append(("Illegal contact < 10%", illegal_contact < 0.10, illegal_contact))
    if "rmse_linear_vel" in results:
        checks.append(("Linear vel RMSE < 0.30", results["rmse_linear_vel"] < 0.30, results["rmse_linear_vel"]))
    if "rmse_yaw_rate" in results:
        checks.append(("Yaw rate RMSE < 0.35", results["rmse_yaw_rate"] < 0.35, results["rmse_yaw_rate"]))
    if "joint_limit_timestep_pct" in results:
        checks.append(("Joint limit < 2% timesteps", results["joint_limit_timestep_pct"] < 2.0, results["joint_limit_timestep_pct"]))
    if "foot_slide_timestep_pct" in results:
        checks.append(("Foot sliding < 10%", results["foot_slide_timestep_pct"] < 10.0, results["foot_slide_timestep_pct"]))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    for desc, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        if isinstance(val, float):
            if val < 1.5:
                val_str = f"{val*100:.1f}%"
            else:
                val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        print(f"  [{status}] {desc}: {val_str}")

    print(f"\n  Passed: {passed}/{total} criteria ({100*passed/max(1,total):.0f}%)")
    if passed >= total * 0.7:
        print(f"  READY FOR STAGE 2 (>=70% criteria met)")
    elif passed >= total * 0.5:
        print(f"  ALMOST THERE (50-70% met), review failures")
    else:
        print(f"  NOT READY FOR STAGE 2 (<50% met)")

if __name__ == "__main__":
    main()
