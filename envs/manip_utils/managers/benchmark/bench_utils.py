import torch
import math

from .relation_monitor import RelationMonitor
from .episode_recorder import EpisodeRecorder
from .benchmark_runtime import BenchmarkRuntime


def _create_subtask_runtime(self):
    from .subtask_runtime import SubtaskRuntime
    self.subtask_runtime = None

    if not self.benchmark_enabled or self.scenario_name is None:
        return

    structure = self.config_manager.load_json("assets/structure.json")

    scenario = structure.get(self.scenario_name)
    if not isinstance(scenario, dict):
        raise ValueError(
            f"Scenario {self.scenario_name!r} was not found "
            "in assets/structure.json"
        )

    annotations = scenario.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(
            f"Scenario {self.scenario_name!r}: 'annotations' " "must be an object"
        )

    self.subtask_runtime = SubtaskRuntime(
        relation_monitor=self.relation_monitor,
        annotations=annotations,
        num_envs=self.num_envs,
    )

def _create_relation_monitor(env, cfg):
    env.relation_monitor = None
    env.relation_recorder = None
    env.benchmark_runtime = None
    # env.benchmark_enabled = bool(getattr(cfg, "benchmark_enabled", False))

    # if env.benchmark_enabled:
    benchmark_task = getattr(cfg, "benchmark_task", None)
    benchmark_output_dir = getattr(cfg, "benchmark_output_dir", "./benchmark_logs")
    benchmark_every_n_steps = int(getattr(cfg, "benchmark_every_n_steps", 1))

    env.relation_monitor = RelationMonitor(
        env=env,
        task_json="assets/tasks/" + benchmark_task + ".json",
        every_n_steps=benchmark_every_n_steps,
    )
    env.relation_recorder = EpisodeRecorder(
        output_dir=benchmark_output_dir,
        topic="/benchmark/rewards",
        task_name=env.relation_monitor.task.get("task", {}).get("name"),
        target_env_id=0,
    )

    control_dt = env.cfg.sim.dt * env.cfg.decimation
    success_grace_steps = math.ceil(env.success_grace_seconds / control_dt)
    env.benchmark_runtime = BenchmarkRuntime(
        env=env,
        success_grace_steps=success_grace_steps,
        debug_print=True,
    )

# def handle_termination_event(env, relation_events: list[dict], event_type: str):
#     ids = [e["env_id"] for e in relation_events if e["type"] == event_type and e.get("value") == 1]
#     if ids:
#         for ev in relation_events:
#             if ev["type"] == event_type and ev.get("value") == 1:
#                 env_id = int(ev["env_id"])
#                 finished_ep_id = int(env.episodes.get_current_episode_id(env_id))
#                 ep_id = f"episode_{finished_ep_id:03d}"
#                 env.relation_recorder._finalize_current(env_id=env_id)
#                 report = env.relation_monitor.build_report(env_id=env_id, episode_id=ep_id)
#                 env.relation_recorder.save_report(report)
#         ids_t = torch.tensor(ids, device=env.device, dtype=torch.long)
#         env.episodes.request_event(event_type, env_ids=ids_t)


def handle_termination_event(self, relation_events: list[dict], event_type: str):
    """
    Handles episode termination events, saves reports and sets termination flags.

    The method filters incoming events by type, extracts the IDs of the
    environments for which the termination condition is satisfied
    (`value == 1`), builds the final reports via the monitor/recorder and
    registers a termination request in EpisodeService.

    Args:
        relation_events (list[dict]): List of event dictionaries.
            Expected structure: {"env_id": int, "type": str, "value": int}.
        event_type (str): Event type key used for filtering and for passing to
            EpisodeService (e.g. "timeout", "success", "manual").

    Note:
        - The save logic runs for all events in the list matching `event_type`.
        - Termination in `self.episodes` is requested in a batch (via a tensor
          of IDs) for optimization.
        - If no matching events are found, the method does nothing.
    """
    # Filter the IDs of environments where the event occurred (value == 1)
    ids = [
        e["env_id"]
        for e in relation_events
        if e["type"] == event_type and e.get("value") == 1
    ]

    if ids:
        # Loop over each completed environment to process and save its data
        for ev in relation_events:
            if ev["type"] == event_type and ev.get("value") == 1:
                env_id = int(ev["env_id"])

                # if env_id == 0 and self.waypoint_tuning is not None:
                #     self.waypoint_tuning.report_result(event_type)

                # Get the current episode ID to name the logs
                sim_time = float(self.sim.current_time)
                finished_ep_id = int(self.episodes.get_current_episode_id(env_id))
                ep_id = f"episode_{finished_ep_id:03d}"

                # Finalize annotations
                episode_payload = self.relation_monitor.get_episode_annotations(
                    env_id=env_id,
                    sim_time=sim_time,
                )

                if self.subtask_runtime is not None:
                    subtask = self.subtask_runtime.get_subtask(env_id)

                    print(
                        "\033[1;93m"
                        f"[SubtaskRuntime]\033[1;36m Episode {event_type}; "
                        f"active subtask: {subtask or 'all subtasks completed'}"
                        "\033[0m"
                    )

                # Finalize recording and save the report
                self.relation_recorder._finalize_current(
                    env_id=env_id,
                    sim_time=sim_time,
                )

                report = self.relation_monitor.build_report(
                    env_id=env_id, episode_id=ep_id
                )
                report["payload"] = episode_payload

                external_episode_id = None
                # if self.save_episodes:
                #     register = event_type == "success"
                #     external_episode_id = self.dataset_logging.on_episode_end(
                #         episode_id=finished_ep_id,
                #         event_type=event_type,
                #         register_in_db=register,
                #         payload=episode_payload,
                #     )
                report["external_episode_id"] = external_episode_id
                self.relation_recorder.save_report(report)

        # Batch-register the termination event in the episode service
        ids_t = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.episodes.request_event(event_type, env_ids=ids_t)

