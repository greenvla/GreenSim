"""Select the active robot asset package from the ``robot`` env var.

Entry points set ``os.environ["robot"]`` from the ``--robot`` CLI argument
before the env modules are imported; everything else imports ConfigManager
from here instead of hardcoding a robot folder.
"""
import importlib
import os

ROBOT = os.environ.get("robot", "green")
ROBOT_FOLDER = f"{ROBOT}"
ROBOT_PKG = f"green_challenge.robots.{ROBOT_FOLDER}"

ConfigManager = importlib.import_module(f"{ROBOT_PKG}.utils.config_manager").ConfigManager
