"""
Singleton Configuration Manager for Robot Configurations

Centralized, globally accessible manager for saving, loading, and managing
multiple named robot configuration files with automatic path resolution and module remapping.

Designed to be used anywhere in the project via:
    from .config_manager import ConfigManager
    cfg = ConfigManager.instance().load_config("robot_hand")
"""

import os
import json
from pathlib import Path
from typing import Optional, Any, List, Union

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.mjcf")

# Import serialization utilities
from ._serializator import (
    save_cfg_json,
    load_cfg_json,
    rename_modules,
    print_package_tree_from_json,
)


class ConfigManager:
    """
    Singleton manager for multiple robot configurations.

    Manages predefined paths for:
    - Robot configuration JSONs (multiple named configs)
    - USD asset files
    - Interface-remapped configuration files

    Provides simplified save/load interface with automatic module remapping.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # --- Resolve paths relative to this file's location ---
        this_file_dir = Path(__file__).parent.resolve()
        dependencies_dir = this_file_dir.parent / "dependencies"

        # if not dependencies_dir.exists():
        #     raise FileNotFoundError(
        #         f"Expected 'dependencies' folder not found at: {dependencies_dir}\n"
        #         f"Current file location: {this_file_dir}\n"
        #         "Please ensure project structure matches:\n"
        #         "project_root/\n"
        #         "├── dependencies/\n"
        #         "│   ├── cfgs/\n"
        #         "│   ├── robots/\n"
        #         "│   └── settings/\n"        
        #     )

        # Set config and USD directories
        self._config_dir = dependencies_dir / "cfgs"
        self._usd_dir = dependencies_dir / "robots"
        self._settings_dir = dependencies_dir / "settings"

        self._make_directory_writable(dependencies_dir)
        for dir in [self._config_dir, self._usd_dir, self._settings_dir]:
            dir.mkdir(parents=True, exist_ok=True)
            self._make_directory_writable(dir)

        # Default config name (for backward compatibility)
        self._default_config_name = "default"

        self._initialized = True

    # --- Properties with Setters ---
    @property
    def settings_dir(self) -> Path:
        """Directory where setting JSONs are stored."""
        return self._settings_dir

    @settings_dir.setter
    def settings_dir(self, path: str | Path):
        self._settings_dir = Path(path).resolve()
        self._settings_dir.mkdir(parents=True, exist_ok=True)
        self._make_directory_writable(self._settings_dir)

    @property
    def config_dir(self) -> Path:
        """Directory where configuration JSONs are stored."""
        return self._config_dir

    @config_dir.setter
    def config_dir(self, path: str | Path):
        self._config_dir = Path(path).resolve()
        self._config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def usd_dir(self) -> Path:
        """Directory where USD robot assets are stored."""
        return self._usd_dir

    @usd_dir.setter
    def usd_dir(self, path: str | Path):
        self._usd_dir = Path(path).resolve()
        self._usd_dir.mkdir(parents=True, exist_ok=True)

    @property
    def default_config_path(self) -> Path:
        """Full path to default robot config JSON."""
        return self.config_path(self._default_config_name)

    @property
    def default_interface_config_path(self) -> Path:
        """Full path to default interface-remapped config JSON."""
        return self.interface_config_path(self._default_config_name)

    @property
    def default_usd_path(self) -> Path:
        """Full path to default exported USD robot asset."""
        return self.get_usd_path()

    # --- Path Helpers ---

    def config_path(self, name: str) -> Path:
        """Get full path for a named config (without extension)."""
        if name.endswith(".json"):
            name = name[:-5]
        return self._config_dir / f"{name}.json"

    def interface_config_path(self, name: str) -> Path:
        """Get full path for a named interface config."""
        if name.endswith(".json"):
            name = name[:-5]
        return self._config_dir / f"{name}.interface.json"

    def get_usd_path(self, usd_name: Optional[str] = None) -> Path:
        """Get full path to a USD asset file."""
        name = usd_name or "robot_exported.usd"
        if not name.endswith(".usd"):
            name += ".usd"
        return self._usd_dir / name

    # --- Public Interface ---

    @classmethod
    def instance(cls) -> "ConfigManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_settings(self, name: str, subdirs: Optional[Union[str, List[str]]] = None) -> dict:
        """
        Load settings from a JSON file in the settings directory.

        Parameters
        ----------
        name : str
            Name of the settings file, with or without the '.json' extension.
        subdirs : str, list of str, or None, optional
            Subdirectory path(s) inside the settings folder.
            Can be a single string (e.g., 'plugins') or a list of strings
            for nested directories (e.g., ['plugins', 'ui', 'themes']).
            Defaults to None (root settings directory).

        Returns
        -------
        dict
            The deserialized JSON content.

        Raises
        ------
        FileNotFoundError
            If the settings file does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        if subdirs is None:
            target_dir = self.settings_dir
        elif isinstance(subdirs, str):
            target_dir = self.settings_dir / subdirs
        else:
            target_dir = self.settings_dir.joinpath(*subdirs)

        filepath = target_dir / name
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_settings_path(
        self,
        name: str,
        subdirs: Optional[Union[str, List[str]]] = None,
    ) -> Path:
        """Returns the full path to a file inside settings."""
        if subdirs is None:
            target_dir = self.settings_dir
        elif isinstance(subdirs, str):
            target_dir = self.settings_dir / subdirs
        else:
            target_dir = self.settings_dir.joinpath(*subdirs)

        filepath = target_dir / name

        if not filepath.is_file():
            raise FileNotFoundError(f"Settings file not found: {filepath}")

        return filepath

    def save_settings(self, data: dict, name: str, subdirs: Optional[Union[str, List[str]]] = None) -> Path:
        """
        Save settings to a JSON file in the settings directory.
        
        Parameters
        ----------
        data : dict
            The data to serialize to JSON.
        name : str
            Name of the settings file (with or without '.json' extension).
        subdirs : str, list of str, or None, optional
            Subdirectory path(s) inside the settings folder.
            Can be a single string or a list of strings for nested directories.
            Defaults to None (root settings directory).
            
        Returns
        -------
        Path
            Path to the saved file.
        """
        # Determine the target directory
        if subdirs is None:
            target_dir = self.settings_dir
        elif isinstance(subdirs, str):
            target_dir = self.settings_dir / subdirs
        else:
            target_dir = self.settings_dir.joinpath(*subdirs)

        filepath = target_dir / name
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved settings '{name}' to: {filepath}")
        return filepath

    def find_asset_paths(self, data: Any) -> List[str]:
        """
        Recursively scan config data and return all 'asset_path' values ending in '.xml'.
        Useful for finding MuJoCo XML asset references.
        Args:
            data: Loaded config structure (dict/list)
        Returns:
            List of asset_path strings (e.g., ["/root/.cache/.../robot_hand.xml"])
        """
        asset_paths = []

        def _scan(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "asset_path" and isinstance(v, str) and v.endswith(".xml"):
                        asset_paths.append(v)
                    else:
                        _scan(v)
            elif isinstance(obj, list):
                for item in obj:
                    _scan(item)

        _scan(data)
        return asset_paths

    @staticmethod
    def _make_directory_writable(path: Path) -> None:
        """
        Recursively make a directory and all its contents readable/writable/executable by everyone.
        Uses chmod 0o777 (rwxrwxrwx) on Unix-like systems.
        Silently skips if permissions cannot be set.
        """
        if not path.exists():
            return

        try:
            # On Unix/Linux/macOS
            if hasattr(os, 'chmod'):
                for root, dirs, files in os.walk(path):
                    root_path = Path(root)
                    # Set dir permissions
                    for d in dirs:
                        dir_path = root_path / d
                        try:
                            dir_path.chmod(0o777)
                        except Exception as ex:
                            pass  # Skip if permission denied or not supported
                    # Set file permissions
                    for f in files:
                        file_path = root_path / f
                        try:
                            file_path.chmod(0o666)  # rw-rw-rw- (no exec needed for files)
                        except Exception as ex:
                            pass
                # Set root dir too
                path.chmod(0o777)
        except Exception as ex:
            # Silent fail — e.g., restricted env
            pass

    def copy_and_remap_assets_in_config(
        self,
        name: str,
        destination_dir: Optional[Path] = None,
        dry_run: bool = False,
        make_writable: bool = True,
    ) -> int:
        """
        Copy asset folders referenced in config to local 'robots' dir and update asset_path
        in the INTERFACE config (not the original).

        For each asset_path like:
            "/root/.cache/models_hub/robot_hand/robot_hand.xml"
        → Copies entire "robot_hand" folder to:
            dependencies/robots/robot_hand/
        → Updates asset_path in .interface.json to be relative to project root:
            "dependencies/robots/robot_hand/robot_hand.xml"

        Leaves original config untouched.

        Args:
            name: Config name to process
            destination_dir: Override destination (default: self.usd_dir)
            dry_run: If True, only print what would be done
            make_writable: If True, set permissive permissions on copied files/folders

        Returns:
            Number of asset paths updated
        """
        config_path = self.config_path(name)
        interface_path = self.interface_config_path(name)

        if not config_path.exists():
            print(f"Config '{name}' not found at {config_path}")
            return 0

        # Load original config to find asset paths (do NOT modify it)
        with open(config_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        asset_paths = self.find_asset_paths(original_data)
        if not asset_paths:
            print(f"No asset_path entries found in '{name}' config.")
            return 0

        dest_dir = destination_dir or self.usd_dir
        updated_count = 0

        # Mapping: old_path → new_local_path (relative to project root)
        asset_mapping = {}

        for old_path_str in asset_paths:
            old_path = Path(old_path_str)
            if not old_path.exists():
                print(f"Asset not found: {old_path}")
                continue

            # Get parent folder name (e.g., "robot_hand")
            asset_folder_name = old_path.parent.name
            local_asset_folder = dest_dir / asset_folder_name
            # Compute path relative to PROJECT ROOT for portability
            # e.g., "dependencies/robots/robot_hand/robot_hand.xml"
            new_asset_path = Path("robots") / asset_folder_name / old_path.name
            asset_mapping[old_path_str] = str(new_asset_path)

            if not dry_run:
                if local_asset_folder.exists():
                    import shutil
                    shutil.rmtree(local_asset_folder)
                    print(f"Deleted existing folder: {local_asset_folder}")

                import shutil
                shutil.copytree(old_path.parent, local_asset_folder)
                print(f"Copied asset folder to: {local_asset_folder}")

                if make_writable:
                    self._make_directory_writable(local_asset_folder)
                    print(f"Set permissive permissions on: {local_asset_folder}")
            else:
                print(f"[DRY RUN] Would copy {old_path.parent} → {local_asset_folder}")

        # Update only the INTERFACE config
        if not interface_path.exists():
            print(f"No interface config found at {interface_path}. Skipping asset path update.")
            return 0

        # Load interface config
        with open(interface_path, "r", encoding="utf-8") as f:
            interface_data = json.load(f)

        # Update asset paths in interface config
        def _update_asset_paths(obj):
            nonlocal updated_count
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "asset_path" and isinstance(v, str) and v in asset_mapping:
                        obj[k] = asset_mapping[v]
                        updated_count += 1
                    else:
                        _update_asset_paths(v)
            elif isinstance(obj, list):
                for item in obj:
                    _update_asset_paths(item)

        _update_asset_paths(interface_data)

        if not dry_run:
            with open(interface_path, "w", encoding="utf-8") as f:
                json.dump(interface_data, f, indent=2, ensure_ascii=False)
            print(f"Updated {updated_count} asset_path(s) in INTERFACE config: {interface_path}")
        else:
            print(f"[DRY RUN] Would update {updated_count} asset_path(s) in {interface_path}")

        return updated_count
    
    def save_config(
        self,
        cfg: Any,
        name: str,
        auto_remap: bool = True,
        source_module: str = "sber_loco_assets",  # ← Default, but configurable
        target_module: str = "green_challenge.robots.green.interface",  # ← Default, but configurable
        remap_assets: bool = True,
        make_assets_writable: bool = True,
    ) -> Path:
        """
        Save named robot configuration to JSON with optional automatic module and asset remapping.

        Args:
            cfg: Configuration object to serialize
            name: Config name (e.g., "robot_hand")
            auto_remap: Whether to automatically create interface-remapped version
            source_module: Source module prefix to replace (e.g., "my_org.assets")
            target_module: Target module prefix for remapping (e.g., "my_project.interface")
            remap_assets: If True, also copy referenced asset folders and update asset_path
            make_assets_writable: If True, set permissive permissions on copied assets

        Returns:
            Path to saved configuration file
        """
        save_path = self.config_path(name)

        # Save original
        save_cfg_json(cfg, str(save_path))
        print(f"Saved config '{name}' to: {save_path}")

        # Optionally create remapped version
        if auto_remap:
            interface_path = self.interface_config_path(name)
            rename_modules(
                in_path=str(save_path),
                old_first=source_module,
                new_first=target_module,
                out_path=str(interface_path),
            )
            print(f"Created interface-remapped config: {interface_path}")

        # Handle asset remapping — only do it once, after interface is created
        if remap_assets:
            print(f"Processing assets for config: {name}")
            self.copy_and_remap_assets_in_config(
                name,
                make_writable=make_assets_writable
            )

        return save_path

    def load_config(
        self,
        name: str,
        use_interface: bool = True,
    ) -> Any:
        """
        Load named robot configuration from JSON, preferring interface-remapped version.

        Args:
            name: Config name (e.g., "robot_hand")
            use_interface: If True, try to load .interface.json version first

        Returns:
            Deserialized configuration object
        """
        base_path = self.config_path(name)

        if use_interface:
            interface_path = self.interface_config_path(name)
            if interface_path.exists():
                print(f"Loading interface config:\n {interface_path}")
                return load_cfg_json(str(interface_path))

        if not base_path.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found at: {base_path}")

        print(f"Loading base config: {base_path}")
        return load_cfg_json(str(base_path))



    def get_cfg_names_by_suffix(self, suffix: str) -> List[str]:
        """
        List all saved configuration names (without extension) that end with the specified suffix.
        Args:
            suffix: The suffix to match filenames against (e.g., "_camera").
        Returns:
            List of config names without extension, sorted alphabetically.
        """
        configs = []
        for f in self._config_dir.glob("*.json"):
            if f.name.endswith(".interface.json"):
                continue  # Skip interface versions
            if f.stem.endswith(suffix):
                name = f.stem  # filename without extension
                configs.append(name)
        return sorted(configs)

    def list_configs(self) -> List[str]:
        """
        List all saved configuration names (without .json extension).

        Returns:
            List of config names
        """
        configs = []
        for f in self._config_dir.glob("*.json"):
            if f.name.endswith(".interface.json"):
                continue  # Skip interface versions
            name = f.stem  # removes .json
            configs.append(name)
        return sorted(configs)

    def delete_config(self, name: str) -> bool:
        """
        Delete a named configuration and its interface version.

        Args:
            name: Config name to delete

        Returns:
            True if at least one file was deleted
        """
        base_path = self.config_path(name)
        interface_path = self.interface_config_path(name)

        deleted = False
        if base_path.exists():
            base_path.unlink()
            print(f"Deleted: {base_path}")
            deleted = True
        if interface_path.exists():
            interface_path.unlink()
            print(f"Deleted: {interface_path}")
            deleted = True

        if not deleted:
            print(f"No config found with name: {name}")

        return deleted

    def print_dependency_tree(self, name: str, module_prefix: str = "sber_loco_assets"):
        """
        Print package dependency tree for a named configuration file.

        Args:
            name: Config name
            module_prefix: Module prefix to filter by
        """
        config_path = self.config_path(name)
        if not config_path.exists():
            print(f"Config '{name}' not found: {config_path}")
            return

        print_package_tree_from_json(str(config_path), first_word=module_prefix)

    def print_asset_paths_in_config(self, name: str) -> None:
        """
        Print all asset_path entries found in a config.

        Args:
            name: Config name to scan
        """
        config_path = self.config_path(name)
        if not config_path.exists():
            print(f"Config '{name}' not found.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        asset_paths = self.find_asset_paths(data)
        if not asset_paths:
            print(f"No asset_path entries found in '{name}'.")
        else:
            print(f"Asset paths in '{name}':")
            for ap in asset_paths:
                print(f"   - {ap}")

    def remap_modules_in_config(
        self,
        name: str,
        old_first: str = "sber_loco_assets",
        new_first: str = "green_challenge.robots.green.interface",
        out_name: Optional[str] = None,
    ) -> int:
        """
        Remap module prefixes in a named configuration file.

        Args:
            name: Input config name
            old_first: Module prefix to replace
            new_first: Replacement module prefix
            out_name: Output config name (if None, overwrites as .interface.json)

        Returns:
            Number of replacements performed
        """
        in_path = self.config_path(name)
        if out_name:
            out_path = self.config_path(out_name)
        else:
            out_path = self.interface_config_path(name)

        return rename_modules(
            in_path=str(in_path),
            old_first=old_first,
            new_first=new_first,
            out_path=str(out_path),
        )

    # --- Backward Compatibility (Optional) ---

    def save_robot_config(
        self,
        cfg: Any,
        config_name: Optional[str] = None,
        auto_remap: bool = True,
        source_module: str = "sber_loco_assets",
        target_module: str = "green_challenge.robots.green.interface",
    ) -> Path:
        """Backward-compatible alias for save_config (uses 'default' if no name given)."""
        name = config_name or self._default_config_name
        return self.save_config(cfg, name, auto_remap, source_module, target_module)

    def load_robot_config(
        self,
        config_name: Optional[str] = None,
        use_interface: bool = True,
    ) -> Any:
        """Backward-compatible alias for load_config."""
        name = config_name or self._default_config_name
        return self.load_config(name, use_interface)

    def load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
# -------------------------------
# Global Access Convenience
# -------------------------------

def get_config_manager() -> ConfigManager:
    """Convenience function to get the singleton instance."""
    return ConfigManager.instance()


# -------------------------------
# Usage Examples / Self-Test
# -------------------------------

if __name__ == "__main__":
    cm = ConfigManager.instance()

    print("Config Directory:", cm.config_dir)
    print("USD Directory:", cm.usd_dir)
    print("Default Config:", cm.default_config_path)
    print("Default Interface Config:", cm.default_interface_config_path)
    print("Default USD Path:", cm.default_usd_path)

    # Example: save multiple configs
    # cm.save_config(robot_hand_cfg, "robot_hand")
    # cm.save_config(my_base_cfg, "mobile_base")

    # Example: load by name
    # robot_hand_cfg = cm.load_config("robot_hand")

    # List all configs
    print("\nSaved Configs:", cm.list_configs())

    # Print tree for a config
    # cm.print_dependency_tree("robot_hand")