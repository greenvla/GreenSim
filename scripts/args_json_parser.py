import argparse
import time
import json

from isaaclab.app import AppLauncher

TYPE_MAP = {
    "int": int,
    "str": str,
    "float": float,
}

def build_parser_from_json(finename:str) -> argparse.ArgumentParser:
    with open(finename, "r", encoding="utf-8") as f:
        cli_args = json.load(f)
    parser = argparse.ArgumentParser()
    for arg in cli_args["arguments"]:
        kwargs = {}
        if "help" in arg:
            kwargs["help"] = arg["help"]
        if "default" in arg:
            kwargs["default"] = arg["default"]
        if "dest" in arg:
            kwargs["dest"] = arg["dest"]
        if "action" in arg:
            kwargs["action"] = arg["action"]
        if "choices" in arg:
            kwargs["choices"] = arg["choices"]
        elif "type" in arg:
            kwargs["type"] = TYPE_MAP[arg["type"]]
        parser.add_argument(*arg["flags"], **kwargs)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(**cli_args.get("app_launcher_defaults", {}))
    return parser

def resolve_optional_feature(cfg):
    if cfg is None:
        return None
    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError(f"Expected [name, enabled] or null, got: {cfg!r}")
    name, enabled = cfg
    if not enabled:
        return None
    return name

def apply_scenario_args(args, scenarios_path: str) -> None:
    if not args.scenario:
        return
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if args.scenario not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {available}")

    scenario_cfg = scenarios[args.scenario]

    if args.scene is None:
        args.scene = scenario_cfg.get("scene")
    if args.benchmark_task is None:
        args.benchmark_task = resolve_optional_feature(scenario_cfg.get("benchmark"))
    if args.policy is None:
        args.policy = resolve_optional_feature(scenario_cfg.get("policy"))
    args.num_episodes = scenario_cfg.get("num_episodes", 10)
