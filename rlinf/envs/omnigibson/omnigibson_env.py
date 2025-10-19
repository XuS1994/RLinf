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
from av.container import Container
from av.stream import Stream
import json
import logging
import os
import sys
import traceback
from inspect import getsourcefile
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import cv2
import gymnasium as gym
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pyvirtualdisplay import Display

# OmniGibson imports
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm, create_module_macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch

# Gello imports
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES

m = create_module_macros(module_path="/mnt/mnt/public/xusi/BEHAVIOR-1K/OmniGibson/omnigibson")
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


display = Display(visible=0, size=(1024,768))
display.start()

__all__ = ["OmnigibsonEnv"]


class OmnigibsonEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics
        self._is_start = True
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self.env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.action_dim = 23
        self.auto_reset = True
        if self.record_metrics:
            self._init_metrics()

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None
        if self.cfg.video_cfg.save_video:
            video_name = str(self.cfg.video_cfg.video_base_dir) + f"/{self.cfg.init_params.task_type}_{self.env._current_episode}.mp4"
            self.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(448, 672),
            )

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.init_params.task_type
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = "R1Pro"
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        # if self.cfg.partial_scene_load:
        #     relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
        #     relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
        #     cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self, actions=None, auto_reset=False) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.

        Returns:
            Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                - extracted_obs (dict): The preprocessed observation dictionary.
                - step_reward (torch.Tensor): The reward for this step.
                - terminations (torch.Tensor): Whether the episode has terminated.
                - truncations (torch.Tensor): Whether the episode was truncated.
                - infos (dict): Additional information about the step.

        Workflow:
            1. Steps the environment with the provided actions and retrieves the next observation,
               termination and truncation flags, and additional info.
            2. If the episode has ended (terminated or truncated), increments the trial counter and
               updates the count of successful trials if the task was completed successfully.
            3. Preprocesses the new observation.
            4. Records metrics and returns the processed results.
        """
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            actions = np.zeros([self.num_envs, self.action_dim])
            self._is_start = False

        raw_obs, step_reward, terminations, truncations, infos = self.env.step(actions, n_render_iterations=1)
        # process obs
        extracted_obs = self._preprocess_obs(raw_obs)

        if isinstance(step_reward, float):
            step_reward = torch.tensor([step_reward], device=self.device)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)

        if terminations or truncations:
            self.n_trials += 1
            if infos["done"]["success"]:
                self.n_success_trials += 1

        infos = self._record_metrics(step_reward, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
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
    def num_envs(self):
        return self.env_args["num_envs"]

    @property
    def device(self):
        return "cuda:0"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.env_args["max_episode_steps"])

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

    def load_task_instance(self, instance_id: int) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    torch.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(torch.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=torch.float32))
                cam_rel_poses.append(torch.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = torch.cat(cam_rel_poses, axis=-1)
        return obs

    def flush_video(self) -> None:
        """
        Flush the video writer.
        """
        self._write_video()

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
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """
        Reset the environment, policy, and compute metrics.
        """
        self.obs = self._preprocess_obs(self.env.reset()[0])
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print stats
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
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

    def _handle_auto_reset(self, past_dones, extracted_obs, infos):
        """Handle automatic reset for environments that have finished."""
        # Reset environments that are done
        reset_obs = self.env.reset()
        if isinstance(reset_obs, tuple):
            reset_obs = reset_obs[0]
        
        # Update observations for reset environments
        for i, is_done in enumerate(past_dones):
            if is_done:
                # Reset metrics for this environment
                self._reset_metrics(env_idx=i)
                # Update observation (this is a simplified version)
                # In practice, you might need to handle this more carefully
                pass
        
        return extracted_obs, infos

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos