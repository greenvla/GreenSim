import torch
from typing import Dict


class Observation:
    def __init__(self, config: Dict, device: torch.DeviceObjType) -> None:
        """
        Observations container class that:
            scales/clip observations according to config
            stacking into observation tensor according to config
        """
        self.device = device
        self.cfg = config
        self.obs_names = self.cfg["order"]
        self.dims = self.parse_attr(self.cfg, "dims", default=None)
        self.scale = self.parse_attr(self.cfg, "scale", default=1.0)
        self.clip = self.parse_attr(self.cfg, "clip", default=[-torch.inf, torch.inf])
        for k in self.clip.keys():  # must be deprecated
            if not isinstance(self.clip[k], list):
                self.clip[k] = [-self.clip[k], self.clip[k]]
        self.num_obs_hist = self.parse_attr(self.cfg, "num_obs_hist", default=1)
        self.obs_buf_size = max(self.num_obs_hist.values())
        self.hist_by_term = self.cfg.get("hist_by_term", False)
        noise_cfg = self.cfg["noise"]
        self.add_noise = noise_cfg is not None and noise_cfg.get("add_noise", False)
        if self.add_noise:
            self.prepare_noise_scale(noise_cfg)

        self.reset()  # set default observsations

    def prepare_noise_scale(self, noise_cfg):
        self.noise_scales = self.parse_attr(noise_cfg, "noise_scales")
        for name in self.noise_scales:
            if name in self.scale:
                self.noise_scales[name] *= self.scale[name]

    def parse_attr(self, config, key, default=None):
        attr = config.get(key, default)
        if isinstance(attr, dict):
            vals = {name: value for name, value in attr.items()}
        else:
            vals = {name: attr for name in self.obs_names}
        return vals

    def process_observation(
        self, obs: torch.Tensor, name: str, inplace: bool = True
    ) -> torch.Tensor:
        if not inplace:
            obs = torch.clone(obs)

        if name in self.scale:
            obs *= self.scale[name]

        if self.add_noise:
            if name in self.noise_scales:
                obs = obs + (2 * torch.rand_like(obs) - 1) * self.noise_scales[name]

        if name in self.clip:
            obs = torch.clip(obs, self.clip[name][0], self.clip[name][1])

        return obs

    def prepare_observations(self, obs_dict: Dict[str, torch.Tensor]):
        prep_obs = []
        for name in self.obs_names:
            observation = obs_dict[name]
            obs = self.process_observation(observation, name)
            prep_obs.append(obs)
        observation = torch.cat(prep_obs, dim=-1)

        if self.obs_buf is None:
            self.obs_buf = observation.repeat(
                self.obs_buf_size
            )  # replicate first observation on init
        else:
            self.obs_buf = torch.cat(
                (self.obs_buf[observation.shape[0] :], observation), dim=-1
            )  # stack latest observation at end

        if self.hist_by_term:
            obs = []
            n_obs = observation.shape[0]
            offset_d = 0
            for o_n in self.obs_names:
                o_d = self.dims[o_n]
                o_h = self.num_obs_hist[o_n]
                o = torch.zeros(o_d * o_h)
                offset_l = n_obs * (
                    self.obs_buf_size - o_h
                )  # take latest for each term
                for h in range(o_h):
                    o[o_d * h : o_d * (h + 1)] = self.obs_buf[
                        offset_l + offset_d + h * n_obs : offset_l
                        + offset_d
                        + h * n_obs
                        + o_d
                    ]
                obs.append(o)
                offset_d += o_d
            obs = torch.cat(obs, dim=-1)
            return obs

        return self.obs_buf

    def check_consistency(self, obs_dict: Dict[str, torch.Tensor]):
        input_obs_names = set(obs_dict.keys())
        missing_names = set(self.obs_names) - input_obs_names
        if len(missing_names) > 0:
            missing_names_str = ", ".join(name for name in missing_names)
            inp_names_str = ", ".join(name for name in input_obs_names)
            declared_names_str = ", ".join(name for name in self.obs_names)

            err_msg = (
                "No names "
                + missing_names_str
                + "\n"
                + "in input observation dict: "
                + inp_names_str
                + "\n"
                + "this names should be specified: "
                + declared_names_str
                + "\n"
            )

            raise AttributeError(err_msg)

    def reset(self):
        self.obs_buf = None
