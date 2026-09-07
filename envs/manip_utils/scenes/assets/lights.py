"""
Light asset configurations for Isaac Lab scenes.
"""
from __future__ import annotations

from isaaclab.sim.spawners import DomeLightCfg, DiskLightCfg
from isaaclab.assets import AssetBaseCfg

# dome_light = AssetBaseCfg(
#     prim_path="/World/Light/DomeLight",
#     spawn=DomeLightCfg(
#         intensity=2000.0,
#         color=(0.75, 0.75, 0.75),
#         enable_color_temperature=True,
#         color_temperature=6500.0,
#     ),
# )

dome_light = AssetBaseCfg(
    prim_path="/World/Light/DomeLight",
    spawn=DomeLightCfg(
        intensity=800.0,
        color=(1.0, 1.0, 1.0),
        enable_color_temperature=True,
        color_temperature=5500.0,
    ),
)

lamp=DiskLightCfg(
        intensity=10000.0,
        color=(0.8, 0.8, 0.8),
        exposure=2.0,
        radius=1.5,
        enable_color_temperature=True,
        color_temperature=6500.0,
        normalize=True,
    )

disk_light = AssetBaseCfg(
    prim_path="/World/Light/DiskLight",
    spawn=lamp,
    init_state=AssetBaseCfg.InitialStateCfg(pos=(1.6, 0.0, 2.5)),
)

disk_light_0 = AssetBaseCfg(
    prim_path="/World/Light/DiskLight_0",
    spawn=lamp,
    init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.6, 0.0, 2.5)),
)

disk_light_1 = AssetBaseCfg(
    prim_path="/World/Light/DiskLight_1",
    spawn=lamp,
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
)


lights_cfgs={"dome_light":dome_light} #"disk_light":disk_light,"disk_light_0":disk_light_0, "disk_light_1":disk_light_1}