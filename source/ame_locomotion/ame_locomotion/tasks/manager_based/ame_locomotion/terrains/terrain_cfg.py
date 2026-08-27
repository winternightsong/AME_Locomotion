# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import *

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, downsampled_scale=0.1, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2, stone_height_max=0.05, stone_width_range=(0.25, 0.5), stone_distance_range=(0.05, 0.25), platform_width=2.0,
            holes_depth=-2.0, border_width=0.25
        ),
        "hf_gaps": HfConcentricGapTerrainCfg(
                    proportion=0.2, gap_width_range=(0.1, 0.5), platform_width=2.0, border_width=0.25, gap_depth=-2.0,
                    ground_width_range=(0.5, 0.5), ground_height_max=0.025
        ),
    },
)
"""Rough terrains configuration."""


S46_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.04, 0.20), step_width=0.30,
            platform_width=2.0, border_width=0.5, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.04, 0.20), step_width=0.30,
            platform_width=2.0, border_width=0.5, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.03, 0.20), platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.01, 0.10), noise_step=0.01,
            downsampled_scale=0.10, border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.05, 0.40), platform_width=2.0, border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.05, 0.40), platform_width=2.0, border_width=0.25,
        ),
        "deep_pits": HfConcentricGapTerrainCfg(
            proportion=0.10, gap_width_range=(0.10, 0.50), ground_width_range=(0.35, 0.65),
            ground_height_max=0.025, gap_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
        "narrow_bridge": HfNarrowBridgeTerrainCfg(
            proportion=0.10, bridge_width_range=(0.35, 0.80), bridge_height_max=0.03,
            pit_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
        "plum_blossom_stakes": HfDoubleColumnStakesTerrainCfg(
            proportion=0.30, stake_height_max=0.06, stake_side_range=(0.32, 0.55),
            stake_gap_range=(0.05, 0.22), column_gap_range=(0.10, 0.22),
            column_jitter=0.04, holes_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
    },
)
"""S46-specific curriculum with pits, narrow bridges, and plum-blossom stakes."""
