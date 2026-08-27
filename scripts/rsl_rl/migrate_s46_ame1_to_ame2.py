#!/usr/bin/env python3
"""Migrate an S46 AME1 checkpoint to an initially behavior-equivalent AME2 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--ame2-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    template = torch.load(args.ame2_template, map_location="cpu", weights_only=False)
    state = source["model_state_dict"]
    template_state = template["model_state_dict"]

    for key, value in template_state.items():
        if key.startswith("global_encoder."):
            state[key] = value.clone()

    query_weight = torch.zeros_like(template_state["query_projector.weight"])
    query_weight[:, 64:] = torch.eye(64, dtype=query_weight.dtype)
    state["query_projector.weight"] = query_weight
    state["query_projector.bias"] = torch.zeros_like(template_state["query_projector.bias"])

    for prefix in ("actor", "critic"):
        key = f"{prefix}.0.weight"
        old_weight = state[key]
        expanded = old_weight.new_zeros(old_weight.shape[0], old_weight.shape[1] + 64)
        expanded[:, 64:] = old_weight
        state[key] = expanded

    migrated = {
        "model_state_dict": state,
        "optimizer_state_dict": None,
        "iter": 0,
        "infos": {
            "migration": "s46_ame1_model36500_to_ame2_behavior_preserving",
            "source_checkpoint": str(args.source),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    print(f"saved: {args.output}")
    print(f"actor.0.weight: {tuple(state['actor.0.weight'].shape)}")
    print(f"critic.0.weight: {tuple(state['critic.0.weight'].shape)}")


if __name__ == "__main__":
    main()
