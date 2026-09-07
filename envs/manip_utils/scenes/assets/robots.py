"""
Robot asset configuration.
"""
from green_challenge.robots.robot_selector import ConfigManager

cm = ConfigManager.instance()

# Load config — will automatically use .interface.json if it exists
robot_cfg = cm.load_config("green_full")