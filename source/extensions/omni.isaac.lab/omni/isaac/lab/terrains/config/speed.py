# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from __future__ import annotations

import omni.isaac.lab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

SPEED_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),    # Sub-terrain size
    border_width=20.0,  # Border around the main terrain (not around subterrains)
    num_rows=10,        # Max difficulty : The difficulty is varied linearly over the number of rows (i.e. along x).
    num_cols=20,        
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
