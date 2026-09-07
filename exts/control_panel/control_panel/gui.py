import omni.ui as ui
from isaacsim.core.utils.viewports import set_camera_view


class ControlPanelGUI:
    def __init__(self, env=None):
        self.env = env
        self._window = None
        self._build_ui()

    def set_env(self, env):
        print(f"[SberRoboticsControl] set_env called, env={env}")
        self.env = env

    def _build_ui(self):
        if self._window is not None:
            self._window.destroy()

        self._window = ui.Window("Sber Robotics Control", width=460, height=160)
        self._window.position_x = 40
        self._window.position_y = 120

        with self._window.frame:
            with ui.VStack(spacing=10):
                with ui.CollapsableFrame("Scene", height=0):
                    with ui.HStack(spacing=6):
                        ui.Button(
                            "Reset Episode",
                            height=32,
                            clicked_fn=self._on_reset,
                        )
                        ui.Button(
                            "Reload Scene",
                            height=32,
                            clicked_fn=self._on_reload_scene,
                        )
                        ui.Button(
                            "Reset Viewport",
                            height=32,
                            clicked_fn=self._on_reset_viewport,
                        )

    def _on_reset(self):
        if self.env is None:
            print("[SberRoboticsControl] env is not available")
            return

        self.env.request_manual_termination()

    def _on_reload_scene(self):
        if self.env is None:
            print("[SberRoboticsControl] env is not available")
            return

        scene_spawner = getattr(self.env, "scene_spawner", None)
        if scene_spawner is None:
            print("[SberRoboticsControl] scene_spawner is not available")
            return

        try:
            scene_spawner.reload_scene()

            viewport_controller = getattr(
                self.env,
                "viewport_camera_controller",
                None,
            )
            if viewport_controller is not None:
                scene_spawner.apply_viewer(viewport_controller)

            print("[SberRoboticsControl] Scene parameters reloaded")

        except Exception as exc:
            print(f"[SberRoboticsControl] reload_scene failed: {exc}")
            return

        self.env.request_manual_termination()

    def _on_reset_viewport(self):
        if self.env is None:
            print("[SberRoboticsControl] env is not available")
            return

        scene_spawner = getattr(self.env, "scene_spawner", None)
        if scene_spawner is None:
            print("[SberRoboticsControl] scene_spawner is not available")
            return

        viewer = getattr(scene_spawner, "viewer_metadata", None)
        if not viewer:
            print("[SberRoboticsControl] viewer_metadata is not available")
            return

        eye = viewer.get("eye")
        lookat = viewer.get("lookat")

        if eye is None or lookat is None:
            print("[SberRoboticsControl] viewer.eye and viewer.lookat are required")
            return

        try:
            set_camera_view(
                eye=list(eye),
                target=list(lookat),
                camera_prim_path="/OmniverseKit_Persp",
            )

            print(
                "[SberRoboticsControl] Viewport reset: "
                f"eye={eye}, target={lookat}"
            )

        except Exception as exc:
            print(f"[SberRoboticsControl] viewport reset failed: {exc}")

    def destroy(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None