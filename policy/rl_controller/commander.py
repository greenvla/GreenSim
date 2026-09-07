import torch
from typing import Dict


class CommandGenerator:
    """Serves to manage environment commands"""

    def __init__(self, config: Dict, num_envs: int, device: torch.DeviceObjType):
        self.device = device
        self.cfg = config
        self.num_envs = num_envs
        self.cmd_names = self.cfg["names"]
        self.num_commands = len(self.cmd_names)

        self.cmd_low = torch.zeros(
            self.num_commands,
            dtype=torch.float,
            device=self.device,
        )
        self.cmd_high = torch.zeros(
            self.num_commands,
            dtype=torch.float,
            device=self.device,
        )
        self.cmd_scales = torch.ones(
            self.num_commands,
            dtype=torch.float,
            device=self.device,
        )

        self._cmd_to_ind = {}
        for ind, name in enumerate(self.cmd_names):
            low, high = self.cfg["ranges"][name]
            scale = self.cfg["scales"][name]
            self._cmd_to_ind[name] = ind
            self.cmd_low[ind] = low
            self.cmd_high[ind] = high
            self.cmd_scales[ind] = scale

        if not hasattr(self.cfg, "zeroable_commands"):
            self.cfg["zeroable_commands"] = []
        if not hasattr(self.cfg, "zero_cmd_thold"):
            self.cfg["zero_cmd_thold"] = 0.0

        self.reset()  # set default commands

    def set_cmd(self, name, value, env_ids=None):
        """Sets values for specified command

        Args:
            name (str): name of command to set values for
            value (torch.Tensor): values to be set
            env_ids (List[int]): Environments ids for which new commands are needed. If None, sets values for all environments
        """
        cmd_ind = self._cmd_to_ind[name]
        value = torch.clip(
            torch.tensor(value, device=self.device),
            self.cmd_low[cmd_ind],
            self.cmd_high[cmd_ind],
        )
        if name in self.cfg["zeroable_commands"]:
            value *= torch.abs(value) > self.cfg["zero_cmd_thold"]
        if env_ids is None:
            self._commands[:, cmd_ind] = value
        else:
            self._commands[env_ids, cmd_ind] = value

    def get_cmd_ind(self, command_names):
        """Maps command names to their indices

        Args:
            names (List[str]): names of commands
        """
        if isinstance(command_names, str):
            return self._cmd_to_ind[command_names]
        return torch.tensor(
            [self._cmd_to_ind[name] for name in command_names], device=self.device
        )

    def get_cmd(self, names=None):
        """Returns values of specified commands

        Args:
            names (List[str]): names of commands to get value of. If None, returns values of all commands
        """
        if names is None:
            return self._commands.clone()
        return self._commands[:, self.get_cmd_ind(names)].clone()

    def reset(self):
        self._commands = torch.zeros(
            self.num_envs,
            self.num_commands,
            dtype=torch.float,
            device=self.device,
        ).clip(self.cmd_low, self.cmd_high)
