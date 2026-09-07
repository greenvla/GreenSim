from __future__ import annotations

from typing import Any, Dict, List, Optional


class BenchmarkRuntime:
    def __init__(self, env, success_grace_steps: int = 8, debug_print: bool = False):
        self.env = env
        self.success_grace_steps = success_grace_steps
        self.debug_print = debug_print
        self.reset()

    def reset(self) -> None:
        self._success_latched = False
        self._success_latched_step: Optional[int] = None
        self._success_emitted = False
        self._latched_success_events: Optional[List[Dict[str, Any]]] = None

    def process(self, sim_step: int, sim_time: float, manual_reset: bool, success_episode: bool) -> None:
        env = self.env

        if not self._success_latched:
            relation_events = env.relation_monitor.update(
                sim_step=sim_step,
                sim_time=sim_time,
                manual_reset=manual_reset,
                success_episode=success_episode,
            )

            if env.subtask_runtime is not None:
                subtask_events = env.subtask_runtime.process(
                    relation_events=relation_events,
                    sim_step=sim_step,
                    sim_time=sim_time,
                    env_ids=[env.relation_recorder.target_env_id],
                )
                env.relation_recorder.record_subtask_events(subtask_events)

            payload = env.relation_recorder.record(relation_events)
            # env.ros_interface.publish_benchmark(*payload)

            if self.debug_print and relation_events:
                for event in relation_events:
                    print(f"\n\n |-{event['type']}--| ", event)

            pending = env.relation_monitor.drain_event_queue(sim_step, sim_time)
            if pending:
                pending_payload = env.relation_recorder.record(pending)
                # env.ros_interface.publish_benchmark(*pending_payload)

            success_events = [
                event for event in relation_events
                if event.get("type") == "success"
            ]

            if success_events:
                self._success_latched = True
                self._success_latched_step = sim_step
                self._latched_success_events = list(relation_events)
            else:
                env.handle_termination_event(relation_events, "timeout")
                env.handle_termination_event(relation_events, "aborted")

        if (
            self._success_latched
            and not self._success_emitted
            and self._success_latched_step is not None
            and sim_step - self._success_latched_step >= self.success_grace_steps
        ):
            pending = env.relation_monitor.drain_event_queue(sim_step, sim_time)
            if pending:
                pending_payload = env.relation_recorder.record(pending)
                # env.ros_interface.publish_benchmark(*pending_payload)

            env.handle_termination_event(self._latched_success_events or [], "success")
            self._success_emitted = True

        if manual_reset:
            env._reset_requested = False