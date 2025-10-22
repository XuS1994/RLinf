# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import json
from typing import Tuple

# Third-party imports
import cv2
import gymnasium as gym
import numpy as np

# OmniGibson imports
import torch
from av.container import Container
from av.stream import Stream
from omegaconf import OmegaConf, open_dict
from omnigibson.macros import gm
from omnigibson.envs import VectorEnvironment
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    TASK_INDICES_TO_NAMES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
# Make sure object states are enabled
gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

from rlinf.envs.libero.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["OmnigibsonEnv"]


class OmnigibsonEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        self.cfg = cfg
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics
        self._is_start = True

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        if self.record_metrics:
            self._init_metrics()

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0

        self._init_env()

        # manually reset environment episode number
        self._video_writer = None
        if self.cfg.video_cfg.save_video:
            video_name = "/mnt/mnt/public/xusi/RLinf-fork-xusi/video/eval/test.mp4"
            self.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(448, 672),
            )

    def _load_tasks_cfg(self):
        with open_dict(self.cfg):
            self.cfg.omnigibson_cfg["task"]["activity_name"] = TASK_INDICES_TO_NAMES[self.cfg.tasks.task_idx]

        with open(self.cfg.tasks.task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[self.cfg.omnigibson_cfg["task"]["activity_name"]]

    def _init_env(self):
        self._load_tasks_cfg()
        self.env = VectorEnvironment(self.cfg.num_envs, OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True))

    def _extract_obs_image(self, raw_obs):
        for _, sensor_data in raw_obs.items():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = v["rgb"].to(torch.uint8)[..., :3].permute(2, 0, 1) / 255.0 # [H, W, C] -> [C, H, W]
                elif "right_realsense_link:Camera:0" in k:
                    right_image = v["rgb"].to(torch.uint8)[..., :3].permute(2, 0, 1) / 255.0 # [H, W, C] -> [C, H, W]
                elif "zed_link:Camera:0" in k:
                    zed_image = v["rgb"].to(torch.uint8)[..., :3].permute(2, 0, 1) / 255.0 # [H, W, C] -> [C, H, W]

        return {
            "images": zed_image, # [C, H, W]
            "wrist_images": torch.stack([left_image, right_image], axis=0), # [N_IMG, C, H, W]
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "images": torch.stack([obs["images"] for obs in extracted_obs_list], axis=0), # [N_ENV, C, H, W]
            "wrist_images": torch.stack([obs["wrist_images"] for obs in extracted_obs_list], axis=0), # [N_ENV, N_IMG, C, H, W]
            "task_descriptions": [
                self.task_description for i in range(self.cfg.num_envs)
            ],
        }
        return obs

    def reset(self):
        raw_obs, infos = self.env.reset()
        obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.cfg.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def step(
        self, actions=None
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raw_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs = self._wrap_obs(raw_obs)

        # if self.video_cfg.save_video:
        #     plot_infos = {
        #         "rewards": step_reward,
        #         "terminations": terminations,
        #         "task": self.task_descriptions,
        #     }
        #     self.add_new_frames(raw_obs, plot_infos)
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations[:] = False
        if self.video_cfg.save_video:
            self._write_video()

        return (
            obs,
            to_tensor(rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_rewards, terminations, truncations, infos = self.step(
                actions
            )
            chunk_rewards.append(step_rewards)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    @property
    def device(self):
        return "cuda:0"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.cfg.max_episode_steps)

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def flush_video(self) -> None:
        """
        Flush the video writer.
        """
        self.video_writer = None

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(
                np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0
            ),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.cfg.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.cfg.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.cfg.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.cfg.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.cfg.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            episode_info = {
                "success": info.get("done", {}).get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            if "success" in info:
                self.success_once[env_idx] = self.success_once[env_idx] | info["success"]
                episode_info["success_once"] = self.success_once[env_idx].clone()
            if "fail" in info:
                self.fail_once[env_idx] = self.fail_once[env_idx] | info["fail"]
                episode_info["fail_once"] = self.fail_once[env_idx].clone()
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        pass
        # reset_state_ids = torch.randint(
        #     low=0,
        #     high=self.total_num_group_envs,
        #     size=(self.num_group,),
        #     generator=self._generator,
        # )
        # self.reset_state_ids = reset_state_ids.repeat_interleave(
        #     repeats=self.group_size
        # ).to(self.device)