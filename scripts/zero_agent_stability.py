"""Finite-horizon zero-action stability diagnostic for an Isaac Lab task."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import ame_locomotion.tasks  # noqa: F401
try:
    import leju_robot  # noqa: F401
except ImportError:
    pass


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    resets = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.long)
    min_root_z = float("inf")
    for step in range(args.steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        _, _, terminated, truncated, _ = env.step(actions)
        resets += (terminated | truncated).long()
        min_root_z = min(min_root_z, env.unwrapped.scene["robot"].data.root_pos_w[:, 2].min().item())
        if (step + 1) % 50 == 0:
            print(
                f"ZERO_PROGRESS step={step + 1} resets={resets.sum().item()} min_root_z={min_root_z:.4f}",
                flush=True,
            )
    print(
        f"ZERO_STABILITY steps={args.steps} envs={args.num_envs} resets={resets.sum().item()} "
        f"envs_with_reset={(resets > 0).sum().item()} max_resets_per_env={resets.max().item()} "
        f"min_root_z={min_root_z:.4f}"
    , flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
