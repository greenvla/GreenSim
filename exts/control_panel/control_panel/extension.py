import asyncio
import omni.kit.app
import omni.ext
import omni.ui as ui
from .gui import ControlPanelGUI


class SberRoboticsControlExtension(omni.ext.IExt):
    
    _instance = None  

    def on_startup(self, ext_id):
        print("[SberRoboticsControl] Extension started")
        self._panel = ControlPanelGUI(env=None)
        SberRoboticsControlExtension._instance = self

    def on_shutdown(self):
        print("[SberRoboticsControl] Extension stopped")
        if self._panel:
            self._panel.destroy()
            self._panel = None
        SberRoboticsControlExtension._instance = None

    def set_env(self, env):
        if self._panel:
            self._panel.set_env(env)

    def on_startup(self, ext_id):
        self._panel = ControlPanelGUI(env=None)
        SberRoboticsControlExtension._instance = self
        asyncio.ensure_future(_dock_window())

async def _dock_window():
    # Wait a few frames for the layout to load
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()
    
    window = ui.Workspace.get_window("Sber Robotics Control")
    property_window = ui.Workspace.get_window("Property")
    
    if window and property_window:
        window.dock_in(property_window, ui.DockPosition.SAME)
