# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass

from .mjcf import spawn_from_mjcf


@configclass
class MJCFFileCfg(FileCfg, converters.MjcfConverterCfg):
    """MJCF file to spawn asset from.

    It uses the :class:`MjfcfConverter` class to create a USD file from MJCF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.
    """

    func: Callable = spawn_from_mjcf
