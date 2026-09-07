from __future__ import annotations
import torch
from isaaclab.utils import configclass
from isaaclab.envs.mdp import terminations
from isaaclab.managers import TerminationTermCfg

def get_termination_by_key(event_key: str):
    """Factory returning a termination function for a specific event type."""
    def termination_func(env):
        return env.episodes.consume_event_mask(event_key)
    return termination_func

@configclass
class NoTerminationsCfg:
    pass


@configclass
class ExampleTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=terminations.time_out)

    relation_timeout = TerminationTermCfg(
        func=get_termination_by_key("timeout"),
        time_out=True,
    )

    relation_success = TerminationTermCfg(
        func=get_termination_by_key("success"),
        time_out=False,
    )

    episode_aborted = TerminationTermCfg(
        func=get_termination_by_key("aborted"),
        time_out=False,
    )