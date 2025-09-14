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

import json
import os

import torch
from omegaconf import DictConfig
# from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from rlinf.config import torch_dtype_from_precision


def get_model_config_and_processor(cfg: DictConfig):
    if cfg.model.model_name == "openvla":
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        from .embodiment.prismatic.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

        model_config = AutoConfig.from_pretrained(cfg.tokenizer.tokenizer_model)

        dataset_statistics_path = os.path.join(
            cfg.tokenizer.tokenizer_model, "dataset_statistics.json"
        )
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(model_config, "norm_stats", norm_stats)
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
    elif cfg.model.model_name == "openvla_oft":
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.prismatic.processing_prismatic import (
            MultiInputPrismaticProcessor as PrismaticProcessorOFT,
        )
        from .embodiment.prismatic.processing_prismatic import PrismaticImageProcessor

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAOFTConfig, PrismaticProcessorOFT)

        model_config = OpenVLAOFTConfig.from_pretrained(
            cfg.tokenizer.tokenizer_model, center_crop=cfg.model.center_crop
        )
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessorOFT.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
    elif cfg.model.model_name == "pi0":
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.configs.policies import PreTrainedConfig
        AutoConfig.register("pi0", PI0Config)
        model_config: PI0Config = PreTrainedConfig.from_pretrained(cfg.tokenizer.tokenizer_model)
        # Pi0 doesn't use traditional tokenizer/image processor
        # It handles preprocessing internally
        input_processor = None
    elif cfg.model.model_name == "openpi":
        # TODO: model_config not used?
        model_config = None
        input_processor = None

    return model_config, input_processor


def get_model(model_path, cfg: DictConfig, override_config_kwargs=None):
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    if cfg.model_name == "openvla":
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        actor_model_config = OpenVLAConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        from .embodiment.openvla_action_model import OpenVLAForRLActionPrediction

        model = OpenVLAForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            hidden_size=cfg.hidden_size,
            unnorm_key=cfg.unnorm_key,
            config=actor_model_config,
            vh_mode=cfg.vh_mode,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            attn_implementation=cfg.attn_implementation,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            trust_remote_code=cfg.trust_remote_code,
        )
    elif cfg.model_name == "openvla_oft":
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.openvla_oft_action_model import OpenVLAOFTForRLActionPrediction

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        override_config_kwargs = cfg
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                setattr(actor_model_config, key, val)

        model = OpenVLAOFTForRLActionPrediction.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            config=actor_model_config,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            trust_remote_code=True,
        )

        # oft add
        model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    elif cfg.model_name == "pi0":
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDatasetMetadata,
        )
        from lerobot.common.policies.factory import make_policy
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.configs.policies import PreTrainedConfig

        from .embodiment.pi0_action_model import Pi0ForRLActionPrediction
        AutoConfig.register("pi0", PI0Config)
        # Load policy configuration from pretrained path
        # actor_model_config: PI0Config = PreTrainedConfig.from_pretrained(model_path)
        actor_model_config: PI0Config = PreTrainedConfig.from_pretrained(model_path)
        actor_model_config.pretrained_path = model_path
        override_config_kwargs = cfg
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                setattr(actor_model_config, key, val)
        model_dir_name = cfg.normalize_name
        dataset_meta = LeRobotDatasetMetadata(
            f"lerobot/{model_dir_name}", root=f"data/{model_dir_name}"
        )
        # TODO: replace the raw make_policy without the metadata. Create the Pi0 wrapper model and set the policy
        model = make_policy(actor_model_config, policy_class=Pi0ForRLActionPrediction, ds_meta=dataset_meta)
        # TODO: solve fsdp bug
        model.model.paligemma_with_expert.replace_gemma_decoder_layer() 
    elif cfg.model_name == "openpi":
        # breakpoint()
        from .embodiment.openpi_action_model import OpenPi0ForRLActionPrediction
        from openpi.training import config as _config
        import openpi.models.model as _model
        import openpi.policies.policy as _policy
        import openpi.shared.download as download
        from openpi.training import checkpoints as _checkpoints
        from openpi.training import config as _config
        import openpi.transforms as transforms
        import safetensors
        # TODO: Only unlock the expert-related parameters
        # TODO: add replace operation on the config by cfg
        # config 
        actor_train_config = _config.get_config("pi0_libero") 
        actor_model_config = actor_train_config.model
        override_config_kwargs = cfg.openpi
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                actor_model_config.__dict__[key] = val
        # load model
        checkpoint_dir = download.maybe_download(str(model_path))
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        model:OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(actor_model_config)
        # train expert only
        if actor_model_config.train_expert_only:
            model.freeze_vlm()
        safetensors.torch.load_model(model, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # fsdp replace 
        # model.paligemma_with_expert.replace_gemma_decoder_layers()
        # load data stats
        data_config = actor_train_config.data.create(actor_train_config.assets_dirs, actor_model_config)
        norm_stats = None
        if norm_stats is None:
            # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
            # that the policy is using the same normalization stats as the original training process.
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load norm stats.")
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        # wrappers
        repack_transforms = transforms.Group()
        default_prompt = None
        model.setup_wrappers(
            transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )
        
    else:
        return None
    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.is_lora:
        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

    if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_dict = torch.load(cfg.ckpt_path)
        model.load_state_dict(model_dict)
    return model
