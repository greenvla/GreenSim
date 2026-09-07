import torch
from typing import List, Dict

class JointManager:
    def __init__(self, env, joint_groups: Dict[str, List[str]]):
        """
        Initialize JointManager with named joint groups.

        Args:
            env: Isaac Lab environment instance.
            joint_groups: Dict like {"legs": ["left_hip...", ...], "fingers": [...]}
        """
        self.env = env.unwrapped
        self.device = self.env.device
        self.num_envs = self.env.num_envs

        action_manager = self.env.action_manager
        active_terms = action_manager.active_terms
        term_dims = action_manager.action_term_dim

        # Build global map: joint_name -> (term_idx, local_index_in_term)
        self._joint_to_location = {}

        for term_idx, term_name in enumerate(active_terms):
            term = action_manager._terms[term_name]
            # Only process terms that have _joint_names (e.g., JointPositionAction)
            if not hasattr(term, '_joint_names'):
                continue
            joint_names = term._joint_names  # list of joint names for this term
            if len(joint_names) != term_dims[term_idx]:
                raise RuntimeError(f"Term '{term_name}': _joint_names length != action_dim")

            for local_idx, joint_name in enumerate(joint_names):
                if joint_name in self._joint_to_location:
                    raise ValueError(f"Joint '{joint_name}' appears in multiple action terms.")
                self._joint_to_location[joint_name] = (term_idx, local_idx)

        # Validate and store group mappings
        self._group_mappings = {}
        for group_name, joint_names in joint_groups.items():
            mapping = []
            for jname in joint_names:
                if jname not in self._joint_to_location:
                    raise ValueError(
                        f"Joint '{jname}' in group '{group_name}' is not controlled by any joint action term."
                    )
                mapping.append(self._joint_to_location[jname])
            self._group_mappings[group_name] = mapping

        # Precompute term slices in the combined action tensor
        self._term_slices = []
        start = 0
        for dim in term_dims:
            self._term_slices.append(slice(start, start + dim))
            start += dim

        self._term_dims = term_dims

        # Initialize internal buffer from current action
        self.reset_buffer()

    def reset_buffer(self):
        """Reset internal action buffer to current env action."""
        self._action_buffer = self.env.action_manager.action.clone()

    def set_group(self, group_name: str, joint_values: List[float]):
        if group_name not in self._group_mappings:
            raise KeyError(f"Group '{group_name}' not registered.")
        mapping = self._group_mappings[group_name]
        if len(joint_values) != len(mapping):
            raise ValueError(f"Group '{group_name}': expected {len(mapping)} values, got {len(joint_values)}.")

        # Instead of zeroing new term_buffers, start from current buffer
        term_updates = {}  # term_idx -> list[(local_idx, value)]
        for value, (term_idx, local_idx) in zip(joint_values, mapping):
            if term_idx not in term_updates:
                term_updates[term_idx] = []
            term_updates[term_idx].append((local_idx, value))

        for term_idx, updates in term_updates.items():
            term_slice = self._term_slices[term_idx]
            # start from existing values in buffer
            updated_term = self._action_buffer[:, term_slice].clone()
            for local_idx, value in updates:
                updated_term[:, local_idx] = value
            self._action_buffer[:, term_slice] = updated_term
    
    def set_joint(self, joint_name: str, value: float):
        if joint_name not in self._joint_to_location:
            raise ValueError(f"Joint '{joint_name}' is not controlled by any joint action term.")
        term_idx, local_idx = self._joint_to_location[joint_name]
        term_slice = self._term_slices[term_idx]
        updated = self._action_buffer[:, term_slice].clone()
        updated[:, local_idx] = value
        self._action_buffer[:, term_slice] = updated

    def set_joints(self, joint_dict: Dict[str, float]):
        term_updates = {}
        for joint_name, value in joint_dict.items():
            if joint_name not in self._joint_to_location:
                raise ValueError(f"Joint '{joint_name}' is not controlled by any joint action term.")
            term_idx, local_idx = self._joint_to_location[joint_name]
            if term_idx not in term_updates:
                term_updates[term_idx] = []
            term_updates[term_idx].append((local_idx, value))

        for term_idx, updates in term_updates.items():
            term_slice = self._term_slices[term_idx]
            updated_term = self._action_buffer[:, term_slice].clone()
            for local_idx, value in updates:
                updated_term[:, local_idx] = value
            self._action_buffer[:, term_slice] = updated_term

    def get_action(self) -> torch.Tensor:
        return self._action_buffer.clone()