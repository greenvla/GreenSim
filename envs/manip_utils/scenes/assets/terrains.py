from __future__ import annotations
from isaaclab.terrains import TerrainImporterCfg

ground_cfg = TerrainImporterCfg(
    prim_path="/World/location",
    terrain_type="plane",
    terrain_generator=None,
    max_init_terrain_level=None,
    collision_group=-1,
    debug_vis=False,
)