import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
import os
import numpy as np
import sys
import accelerate
from omegaconf import DictConfig
sys.path.append('/mnt/mnt/public/liuzhihao/RLinf_0828')
from rlinf.models.embodiment.pi0_action_model import Pi0ForRLActionPrediction
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.policies import PreTrainedConfig
from rlinf.models.embodiment.pi0_action_model import Pi0ForRLActionPrediction
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
import functools
import os

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import AutoModelForCausalLM

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import (
    get_fsdp_wrap_policy,
    init_fn,
)
from rlinf.utils.utils import clear_memory


# model
model_name = "pi0" # pi0, simplemodel

# TODO: 恢复model.py里面的bf16->float32
import random
import torch
import numpy as np
def set_seed(seed):
    """
    为所有相关库设置一个固定的随机种子，确保可复现性。
    """
    torch.manual_seed(seed)
    # 如果使用GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # 确保cuDNN的确定性，但这可能会影响性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def main():
    # INIT DDP
    # TODO: FOR DEBUG
    # import os
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29500"
    # os.environ["WORLD_SIZE"] = "1"
    # import torch.distributed as dist
    # def init_distributed():
    #     """Initialize distributed training"""
    #     if not dist.is_initialized():
    #         dist.init_process_group(
    #             backend="nccl" if torch.cuda.is_available() else "gloo",
    #             init_method="env://",
    #             world_size=int(os.environ.get("WORLD_SIZE", 1)),
    #             rank=int(os.environ.get("RANK", 0))
    #         )
    # init_distributed()
    # TODO: FOR DEBUG
    # PI0-MODEL
    accelerator = Accelerator()
    if model_name == "pi0":
        pretrained_path = "/mnt/mnt/public/liuzhihao/megatron-infinigence-rl-chenk/pretrained_model/"
        policy_cfg: PI0Config = PreTrainedConfig.from_pretrained(pretrained_path)
        policy_cfg.pretrained_path = pretrained_path
        policy_cfg.num_steps = 3
        policy_cfg.noise_level = 0.5
        policy_cfg.train_expert_only = True
        kwargs = {"config": policy_cfg}
        dataset_meta = LeRobotDatasetMetadata(
            f"lerobot/libero_spatial_image", root=f"data/libero_spatial_image"
        )
        model: Pi0ForRLActionPrediction = make_policy(policy_cfg, policy_class=Pi0ForRLActionPrediction, ds_meta=dataset_meta)
        model.model.paligemma_with_expert.replace_gemma_decoder_layer() # TODO: solve fsdp bug
        # model = model.to(torch.bfloat16)
        # FSDP
        def should_wrap(module):
            # return False
            # TODO: zhihao: add PaliGemmaForConditionalGeneration to the should_wrap function
            # TODO cannot import name 'PaliGemmaForConditionalGeneration' from 'transformers' in transformers 4.40.1
            from transformers import PaliGemmaForConditionalGeneration
            from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
            from lerobot.common.policies.normalize import Normalize, Unnormalize
            if isinstance(module, PaliGemmaForConditionalGeneration):
                return True
            elif isinstance(module, GemmaDecoderLayer): 
                return True
            else:
                return False

        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=should_wrap)

        model = FSDP(
            model,
            # param_init_fn=init_fn,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            device_id=int(accelerator.process_index),
            sharding_strategy=sharding_strategy,  # zero3
            # mixed_precision=mixed_precision,
            sync_module_states=True,
        )
    elif model_name == "simplemodel":
        model = SimpleModel()
        model = FSDP(
            model,
            device_id=int(accelerator.process_index),
            sharding_strategy=ShardingStrategy.NO_SHARD,  # zero3
            sync_module_states=True,
        )
    model = model.train()
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-1, weight_decay=0.01)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 伪训练过程
    train_pi0_policy(model, optimizer, scheduler, accelerator, num_epochs=10)

def create_dummy_batch(batch_size=4, device="cuda"):
    global accelerator
    """创建伪训练数据批次"""
    # 模拟观测数据
    if model_name == "pi0":
        batch = {
            "observation.images.image": torch.randn(batch_size, 3, 224, 224, device=device),
            "observation.images.wrist_image": torch.randn(batch_size, 3, 224, 224, device=device),
            "observation.state": torch.randn(batch_size, 8, device=device),  # 7维机器人状态
            "action": torch.randn(batch_size, 5, 7, device=device),  # 5步动作序列，每步7维
            "task": ["pick up the red block"] * batch_size,
            "chains": torch.randn(batch_size, 4, 50, 32, device=device),
            "denoise_inds": torch.randint(0, 3, (batch_size,), device=device),
            "prev_logprobs": torch.randn(batch_size, 3, 5, 7, device=device),
            "reward": torch.randn(batch_size, device=device),
            "done": torch.zeros(batch_size, dtype=torch.bool, device=device),
            }

    elif model_name == "simplemodel":
        batch = {
            "x": torch.randn(batch_size, 10, device=device),
            "reward": torch.randn(batch_size, device=device),
        }
    
    # for key, value in batch.items():
    #     if key not in ["task", "denoise_inds"]:
    #         batch[key] = value.to(torch.bfloat16)
    
    return batch

def compute_ppo_loss(model, batch, advantages, accelerator, clip_ratio=0.2):
    """计算PPO损失"""
    # 计算新的log概率
    for key, value in batch.items():
        if key not in ["task", "denoise_inds"]:
            batch[key] = batch[key] * (accelerator.process_index + 1)

    if model_name == "pi0":
        output_dict = model(batch, mode="compute_logprob")
        new_log_probs = output_dict["logprobs"]
        old_log_probs = new_log_probs.detach()
        # 计算概率比率
        ratio = torch.exp(new_log_probs.sum(dim=[1,2]) - old_log_probs.sum(dim=[1,2]))
    elif model_name == "simplemodel":
        new_log_probs = model(batch["x"])
        old_log_probs = new_log_probs.detach()
        ratio = torch.exp(new_log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))
    
    # 计算PPO损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


def train_pi0_policy(model, optimizer, scheduler, accelerator, num_epochs=10):
    """PI0策略的伪训练过程"""
    print("开始PI0策略训练...")
    
    model.train()

    for step in range(5): 
        # 创建伪数据批次
        batch = create_dummy_batch(batch_size=4, device=accelerator.device)
        
        # 模拟计算优势和回报（实际中这些来自环境交互）
        advantages = torch.randn_like(batch["reward"]) * 0.1  # 模拟优势
        
        # 计算PPO损失
        loss = compute_ppo_loss(
            model, batch, advantages, accelerator
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        # grad_norm = FSDP.clip_grad_norm_(model, max_norm=1.0)
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm = model.clip_grad_norm_(max_norm=1.0)
        mean_param = 0.0
        for name, param in model.named_parameters():
            mean_param += param.mean()
        print(f"step: {step} rank: {accelerator.process_index} grad_norm: {grad_norm} mean_param: {mean_param}")
        optimizer.step()




if __name__ == "__main__":
    main()