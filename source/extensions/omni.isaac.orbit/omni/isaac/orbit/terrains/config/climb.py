# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from __future__ import annotations

import omni.isaac.orbit.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.02, 0.28),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.02, 0.28),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        )
    },
)
"""Rough terrains configuration."""
