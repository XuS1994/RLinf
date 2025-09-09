import importlib
import os
import sys

import cv2
import torch
import yaml

# TODO: envs is referenced from RoboTwin repo, need to be modified
from envs import *

sys.path.append('./')
sys.path.append('./policy/RDT/')
from multiprocessing import Array, Event, Semaphore, Value

import gymnasium as gym
import numpy as np
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

'''
输入值:
actions[step,14]
完整返回值:
images[step,3,3,480,640]
states[step,action_dim]
'''

DEBUG = True

def Debug_print(process_id, msg):
    if DEBUG:
        print(f"[process {process_id}] {msg}")

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as e:
        print(f"Error in class_decorator: {e}")
        raise SystemExit("No Task")
    return env_instance

def worker(
    process_id: int,
    task_name: str,
    args: dict,
    seed_id: Value,
    actions: Array,
    results: Array,
    action_input_sem: Semaphore,
    result_output_sem: Semaphore,
    reset_event: Event,
    ):
    global GPU_ID
    NUM_PROCESSES = args['n_envs']
    HORIZON = args['horizon']
    ACTION_DIM = args['action_dim']
    RESULT_SIZE = args['result_size']
    # INPUT_SIZE = args['input_size']

    envs_finish = False
    def filling_up(return_poses):
        if len(return_poses) < HORIZON:
            new_poses = np.empty((HORIZON, *return_poses.shape[1:]))
            new_poses[:len(return_poses)] = return_poses
            new_poses[len(return_poses):] = return_poses[-1]
            return new_poses
        return return_poses

    task = class_decorator(task_name)

    valid_seed = False
    while not valid_seed:
        try:
            with seed_id.get_lock():
                now_seed = seed_id.value
                seed_id.value += 1
                if seed_id.value >= 30:
                    seed_id.value %= 30
            print(process_id, f"now_seed: {now_seed}")
            task.setup_demo(now_ep_num = now_seed, seed = now_seed, ** args)
            valid_seed = True
        except Exception as e:
            print(f"Error in process {process_id} during setup_demo with seed {now_seed}: {e}")

    # 初始化成功后需要返回初始场景
    action_input_sem.acquire()
    task.run_steps = 0 #用于记录当前运行到第几步, 大于每一次环境的最大执行部步数就结束, shoe_place中是450步
    prev_obs_venv = task.get_obs()
    return_pose = np.array([task.get_return_pose()])
    return_pose = filling_up(return_pose)
    prev_obs_venv = [prev_obs_venv]
    image_return = update_obs(prev_obs_venv[-1])[0] + update_obs(prev_obs_venv[-1])[0]
    state_return = update_obs(prev_obs_venv[-1])[1]
    image_return = np.array(image_return)
    state_return = np.array(state_return)

    result = np.concatenate([image_return.flatten(), state_return.flatten(), np.array([0]), np.array([0]), np.array([0]), return_pose.flatten()])
    print(process_id, "task init success!")
    print(f"image_return.shape {image_return.shape}")
    print(f"state_return.shape {state_return.shape}")
    print(f"return_pose.shape {return_pose.shape}")
    print(f"len(result) {len(result)}")
    print(f"len(results) {len(results)}")
    print(RESULT_SIZE)
    results[RESULT_SIZE*process_id:(process_id+1)*RESULT_SIZE] = result
    result_output_sem.release()

    while True:
        # 接收到action_input_event信号量,表示要actions更新了
        action_input_sem.acquire()
        # print(process_id, f"read from shared buffer")
        numpy_actions = np.frombuffer(actions.get_obj()).reshape(NUM_PROCESSES, HORIZON, ACTION_DIM)
        input_actions = numpy_actions[process_id]

        if reset_event.is_set():
            valid_seed = False

            while not valid_seed:
                try:
                    with seed_id.get_lock():
                        now_seed = seed_id.value
                        seed_id.value += 1
                        if seed_id.value >= 30:
                            seed_id.value %= 30
                    print(process_id, f"now_seed: {now_seed}")
                    task.setup_demo(now_ep_num = now_seed, seed = now_seed, ** args)
                    valid_seed = True
                except Exception as e:
                    print(f"Error in process {process_id} during setup_demo with seed {now_seed}: {e}")
            task.run_steps = 0 #用于记录当前运行到第几步, 大于每一次环境的最大执行部步数就结束, shoe_place中是450步
            task.reward_step = 0 # 对应获取reward阶段
            envs_finish = False
            prev_obs_venv = task.get_obs()
            prev_obs_venv = [prev_obs_venv]
            image_return = update_obs(prev_obs_venv[-1])[0] + update_obs(prev_obs_venv[-1])[0]
            state_return = update_obs(prev_obs_venv[-1])[1]
            image_return = np.array(image_return)
            state_return = np.array(state_return)
            task.reward.initialize()

            image_return = np.array(image_return)
            state_return = np.array(state_return)

            result = np.concatenate([image_return.flatten(), state_return.flatten(), np.array([0]), np.array([0]), np.array([1]), return_pose.flatten()])
            results[RESULT_SIZE*process_id:RESULT_SIZE*(process_id+1)] = result
            result_output_sem.release()

            print(process_id, "reset success!")
            continue

        '''
        obs_venv,
        reward_venv,
        terminated_venv,
        truncated_venv,
        info_venv,#暂时没用, 不返回, 后续可以补充
        '''
        obs_venv,reward_venv, terminated_venv, _, return_poses = task.gen_dense_reward_once(input_actions)
        # print(process_id, f"step once success!")
        # TODO something return_poses is [1,6], sometimes is [2,6]
        return_poses = return_poses[0:1,:]
        return_poses = filling_up(return_poses)
        if len(obs_venv) > 1: #表示没有运行结束/成功
            image_return = update_obs(obs_venv[-2])[0] + update_obs(obs_venv[-1])[0]
            state_return = update_obs(obs_venv[-1])[1]
        # 按顺序编码
        # 判断是否要进行新的环境了
        if terminated_venv[0] == 1:
            envs_finish = True
            print(process_id, "task terminated!")

        # print(task.run_steps, now_seed)
        if envs_finish:
            valid_seed = False

            while not valid_seed:
                try:
                    with seed_id.get_lock():
                        now_seed = seed_id.value
                        seed_id.value += 1
                        if seed_id.value >= 30:
                            seed_id.value %= 30
                    print(process_id, f"now_seed: {now_seed}")
                    task.setup_demo(now_ep_num = now_seed, seed = now_seed, ** args)
                    valid_seed = True
                except Exception as e:
                    print(f"Error in process {process_id} during setup_demo with seed {now_seed}: {e}")
            task.run_steps = 0 #用于记录当前运行到第几步, 大于每一次环境的最大执行部步数就结束, shoe_place中是450步
            task.reward_step = 0 # 对应获取reward阶段
            envs_finish = False
            prev_obs_venv = task.get_obs()
            prev_obs_venv = [prev_obs_venv]
            image_return = update_obs(prev_obs_venv[-1])[0] + update_obs(prev_obs_venv[-1])[0]
            state_return = update_obs(prev_obs_venv[-1])[1]
            image_return = np.array(image_return)
            state_return = np.array(state_return)
            task.reward.initialize()

        image_return = np.array(image_return)
        state_return = np.array(state_return)

        result = np.concatenate([image_return.flatten(), state_return.flatten(), reward_venv.flatten(),terminated_venv.flatten(), np.array([0]), return_poses.flatten()])
        print(process_id, "task final success!")
        print(f"image_return.shape {image_return.shape}")
        print(f"state_return.shape {state_return.shape}")
        print(f"return_pose.shape {return_poses.shape}")
        print(f"len(result) {len(result)}")
        print(f"len(results) {len(results)}")
        print(RESULT_SIZE)
        results[RESULT_SIZE*process_id:RESULT_SIZE*(process_id+1)] = result
        result_output_sem.release()

def update_obs(observation):
    imgs = []
    imgs.append(observation['observation']['head_camera']['rgb'][:,:,::-1])
    imgs.append(observation['observation']['right_camera']['rgb'][:,:,::-1])
    imgs.append(observation['observation']['left_camera']['rgb'][:,:,::-1])
    state = observation['joint_action']["vector"]
    return imgs, state

class RoboTwin(gym.Env):
    def __init__(self, cfg, rank, world_size, record_metrics=True):
        # 从配置中获取参数
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.record_metrics = record_metrics
        self._is_start = True
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self.env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        if self.record_metrics:
            self._init_metrics()

        self.task_name = "place_shoe"
        self.n_envs = self.num_envs
        self.horizon = getattr(cfg, 'horizon', 1)
        self.action_dim = 14
        self.root_path = "/mnt/public/xusi/merge_repo/RLinf_RoboTwin/"
        self.configs_path = "/mnt/public/xusi/merge_repo/RLinf_RoboTwin/task_config/"
        head_camera_type = 'D435'
        seed = 1
        rdt_step = 10
        with open(os.path.join(self.root_path, f'task_config/{self.task_name}.yml'), 'r', encoding='utf-8') as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        embodiment_type = args.get('embodiment')
        embodiment_config_path = os.path.join(self.configs_path, '_embodiment_config.yml')

        with open(embodiment_config_path, 'r', encoding='utf-8') as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(self.configs_path + '_camera_config.yml', 'r', encoding='utf-8') as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        args['head_camera_h'] = _camera_config[head_camera_type]['h']
        args['head_camera_w'] = _camera_config[head_camera_type]['w']
        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]['file_path']
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, 'config.yml')
            with open(robot_config_file, 'r', encoding='utf-8') as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        if len(embodiment_type) == 1:
            args['left_robot_file'] = os.path.join(self.root_path, get_embodiment_file(embodiment_type[0]))
            args['right_robot_file'] = os.path.join(self.root_path, get_embodiment_file(embodiment_type[0]))
            args['dual_arm_embodied'] = True
        elif len(embodiment_type) == 3:
            args['left_robot_file'] = get_embodiment_file(embodiment_type[0])
            args['right_robot_file'] = get_embodiment_file(embodiment_type[1])
            args['embodiment_dis'] = embodiment_type[2]
            args['dual_arm_embodied'] = False
        else:
            raise "embodiment items should be 1 or 3"

        args['left_embodiment_config'] = get_embodiment_config(args['left_robot_file'])
        args['right_embodiment_config'] = get_embodiment_config(args['right_robot_file'])

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + '_' + str(embodiment_type[1])

        # output camera config
        # print('============= Config =============\n')
        # print('Messy Table: ' + str(args['messy_table']))
        # print('Random Texture: ' + str(args['random_texture']))
        # print('Head Camera Config: '+ str(args['head_camera_type']) + f',' + str(args['collect_head_camera']))
        # print('Wrist Camera Config: '+ str(args['wrist_camera_type']) + f',' + str(args['collect_wrist_camera']))
        # print('Embodiment Config:: '+ str(args['embodiment']))
        # print('\n=======================================')

        args['embodiment_name'] = embodiment_name
        args['expert_seed'] = seed

        args['rdt_step'] = rdt_step
        args['save_path'] += f"/{self.task_name}_reward"

        # global NUM_PROCESSES, HORIZON, ACTION_DIM, RESULT_SIZE, INPUT_SIZE
        args['n_envs'] = self.num_envs
        args['horizon'] = self.horizon
        args['action_dim'] = 14

        self.NUM_IMAGES = 6
        self.IMAGE_SHAPE = (240, 320, 3)  # 每张图像的形状
        self.STATE_SHAPE = (1, 14)  # 状态向量的形状
        self.TARGET_SHAPE = (self.horizon, 6)  # 目标物体的xyz + 目标位置xyz

        self.IMAGE_SIZE = np.prod(self.IMAGE_SHAPE)  # 每张图像的大小
        self.STATE_SIZE = np.prod(self.STATE_SHAPE)  # 状态向量的大小
        self.TARGET_SIZE = np.prod(self.TARGET_SHAPE) # 目标向量的大小

        args['result_size'] = int(self.NUM_IMAGES * self.IMAGE_SIZE + self.STATE_SIZE + 3 + self.TARGET_SIZE)  # 输出大小
        args['input_size'] = int(self.horizon*14)   # 输入大小

        self.args = args
        # 多进程的线程数
        self.n_envs = self.num_envs

        self.process = []

        self.seed = 0
        self.input_sem = None
        self.output_sem = None
        self.reset_event = None
        self.share_seed = None
        self.share_actions = None
        self.share_results = None

        self.init_process()

    @property
    def num_envs(self):
        return self.env_args['num_envs']

    @property
    def device(self):
        return "cpu"  # RoboTwin使用CPU

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.all_reset_state_ids = torch.randperm(
            self.total_num_group_envs, generator=self._generator
        ).to(self.device)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        # TODO check if this is needed
        pass

    def _extract_obs_image(self, raw_obs):
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        obs_image = obs_image.permute(0, 3, 1, 2)  # [B, C, H, W]
        extracted_obs = {"images": obs_image, "task_descriptions": self.instruction}
        return extracted_obs

    def _calc_step_reward(self, info):
        reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.env.device
        )  # [B, ]
        reward += info["is_src_obj_grasped"] * 0.1
        reward += info["consecutive_grasp"] * 0.1
        reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

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

    # 初始化成功后需要返回初始场景
    def init_process(self):
        self.context = mp.get_context("spawn")
        self.input_sem = self.context.Semaphore(0)
        self.output_sem = self.context.Semaphore(0)
        # reset()使用信号控制
        self.reset_event = self.context.Event()

        self.share_seed = self.context.Value('i', self.seed)
        self.share_actions = self.context.Array('d', self.args['n_envs'] * self.args['input_size'])
        self.share_results = self.context.Array('d', self.args['n_envs'] * self.args['result_size'])
        # self.temp_worker = worker(0, self.task_name, self.args, self.share_seed, self.share_actions, self.share_results, self.input_sem, self.output_sem, self.reset_event)
        for i in range(self.n_envs):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
            p = self.context.Process(target=worker, args=(i, self.task_name, self.args, self.share_seed, self.share_actions, self.share_results, self.input_sem, self.output_sem, self.reset_event), daemon=True)
            self.process.append(p)
            p.start()

        for _ in range(self.n_envs):
            self.input_sem.release()

        for _ in range(self.n_envs):
            self.output_sem.acquire()

        results = np.frombuffer(self.share_results.get_obj()).reshape(self.n_envs, self.args['result_size'])
        results = self.de_initialize_result(results)
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.transform(results)
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def step(self, actions = None):
        if actions is None:
            actions = np.zeros((self.n_envs, self.horizon, self.action_dim))
        print(f"actions.shape: {actions.shape}")
        self.share_actions[:] = actions.flatten()

        # 释放信号量, 表示actions更新了
        for i in range(self.n_envs):
            self.input_sem.release()

        # 等待所有envs都完成了输出
        for i in range(self.n_envs):
            self.output_sem.acquire()

        results = np.frombuffer(self.share_results.get_obj()).reshape(self.n_envs, self.args['result_size'])
        results = self.de_initialize_result(results)
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.transform(results)
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def reset(self):
        self.reset_event.set()
        # 释放信号量, 表示actions更新了
        for i in range(self.n_envs):
            self.input_sem.release()

        # 等待所有envs都完成了输出
        for i in range(self.n_envs):
            self.output_sem.acquire()

        self.reset_event.clear()
        return

    # def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    #     """Reset the environment and return initial observation and info."""
    #     if options is None:
    #         options = {}

    #     # 如果这是第一次调用或需要初始化进程
    #     if self.input_sem is None:
    #         obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.init_process()
    #         self._is_start = False
    #         self._elapsed_steps = 0
    #         # 返回第一个环境的观察
    #         return obs_venv[0], {"return_poses": info_venv[0]}

    #     # 常规重置
    #     self.reset_event.set()
    #     # 释放信号量, 表示actions更新了
    #     for i in range(self.n_envs):
    #         self.input_sem.release()

    #     # 等待所有envs都完成了输出
    #     for i in range(self.n_envs):
    #         self.output_sem.acquire()

    #     self.reset_event.clear()
    #     self._elapsed_steps = 0

    #     # 获取重置后的观察
    #     results = np.frombuffer(self.share_results.get_obj()).reshape(self.n_envs, self.args['result_size'])
    #     results = self.de_initialize_result(results)
    #     obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.transform(results)

    #     # 返回第一个环境的观察
    #     return obs_venv[0], {"return_poses": info_venv[0]}

    # def step(self, actions):
    #     """Execute actions and return observation, reward, terminated, truncated, info."""
    #     if self.is_start:
    #         # 如果是第一次调用，先重置环境
    #         obs, info = self.reset()
    #         self._is_start = False
    #         terminated = np.zeros(self.num_envs, dtype=bool)
    #         truncated = np.zeros(self.num_envs, dtype=bool)
    #         reward = np.zeros(self.num_envs, dtype=np.float32)
    #         return obs, reward[0], terminated[0], truncated[0], info

    #     # 确保actions是正确的形状
    #     if len(actions.shape) == 2:
    #         actions = actions[np.newaxis, :]  # 添加batch维度
    #     elif len(actions.shape) == 1:
    #         actions = actions.reshape(1, self.horizon, -1)

    #     self.share_actions[:] = actions.flatten()

    #     # 释放信号量, 表示actions更新了
    #     for i in range(self.n_envs):
    #         self.input_sem.release()

    #     # 等待所有envs都完成了输出
    #     for i in range(self.n_envs):
    #         self.output_sem.acquire()

    #     results = np.frombuffer(self.share_results.get_obj()).reshape(self.n_envs, self.args['result_size'])
    #     results = self.de_initialize_result(results)
    #     obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.transform(results)

    #     self._elapsed_steps += 1

    #     # 检查是否达到最大步数
    #     truncated_venv = [t or (self._elapsed_steps >= self.max_steps) for t in truncated_venv]

    #     # 返回第一个环境的结果
    #     return (obs_venv[0],
    #             reward_venv[0],
    #             terminated_venv[0],
    #             truncated_venv[0],
    #             {"return_poses": info_venv[0]})

    def transform(self, results):
        # TODO resize image to 224x224
        size = (224,224)
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode('.jpg', img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img
        def resize_img(img,size):
            return cv2.resize(img, size)
        obs_venv = {"images": [], "state": [], "task_descriptions": []}
        reward_venv = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )  # [B, ]
        terminated_venv = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        truncated_venv = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        info_venv = []
        for i in range(self.n_envs):
            result = results[i]
            imgs = result['imgs']
            imgs = [resize_img(img,size) for img in imgs]
            imgs = [jpeg_mapping(img) for img in imgs]
            imgs = np.array(imgs)

            # TODO output is 6 3 224 224, we just use first images
            obs_venv["images"].append(torch.from_numpy(imgs).to(self.device)[0])
            obs_venv["state"].append(torch.from_numpy(result['state']).to(self.device))
            obs_venv["task_descriptions"].append("")
            reward_venv[i] = torch.from_numpy(result['reward']).to(self.device)
            terminated_venv[i] = torch.from_numpy(result['terminated']).to(self.device)
            truncated_venv[i] = torch.from_numpy(result['truncated']).to(self.device)
            info_venv.append(result['return_poses'])
            print(f"state: {result['state']} and result keys: {result.keys()}")
        obs_venv["images"] = torch.stack(obs_venv["images"]).permute(0,3,1,2)
        obs_venv["state"] = torch.stack(obs_venv["state"])
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            print(f"chunk_actions: {chunk_actions.shape}")
            print(f"actions: {actions.shape}")
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions
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

    def de_initialize_result(self, results):
        conds = []
        for i in range(self.n_envs):
            cond = {}
            start = 0
            result = results[i]
            cond["imgs"] = result[start:start + self.NUM_IMAGES * self.IMAGE_SIZE].reshape(self.NUM_IMAGES, *self.IMAGE_SHAPE)
            start += self.NUM_IMAGES * self.IMAGE_SIZE
            cond["state"] = result[start:start + self.STATE_SIZE].reshape(self.STATE_SHAPE)
            start += self.STATE_SIZE
            cond["reward"] = result[start:start + 1]
            start += 1
            cond["terminated"] = result[start:start + 1]
            start += 1
            cond["truncated"] = result[start:start + 1]
            start += 1
            cond["return_poses"] = result[start:start + self.TARGET_SIZE].reshape(self.TARGET_SHAPE)
            start += self.TARGET_SIZE
            conds.append(cond)
        return conds

    def clear(self):
        # 结束所有的当前进程
        for i in range(self.n_envs):
            if self.process[i].is_alive():
                self.process[i].terminate()
                # 等待进程完全结束
                self.process[i].join()
                # 释放占用内存
                self.process[i].close()
        self.seed = self.share_seed.value
        if self.seed > 3000:
            self.seed = 0
        self.process = []
        print("main", "clear all processes success!")

if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = 'place_shoe'
    n_envs = 2
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10
    robotwin = RoboTwin(task_name, n_envs, horizon, steps)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        prev_obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = robotwin.init_process()
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim)*0.05
            actions = np.clip(actions,0, 1)
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = robotwin.step(actions)
            print("main", f"step: {step}")
            print("main", f"reward_venv: {reward_venv}")

            if step%10 == 0:
                robotwin.reset()
            if terminated_venv[0] == 1:
                print("main", f"terminated_venv: {terminated_venv}")
            if truncated_venv[0] == 1:
                print("main", f"truncated_venv: {truncated_venv}")
            # print("main", f"info_venv: {info_venv}")
        robotwin.clear()
