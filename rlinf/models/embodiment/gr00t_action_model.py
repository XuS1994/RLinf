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
import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.modules.value_head import ValueHead

logger = logging.getLogger(__name__)


# TODO(lx): It's copied from openpi. Current env doesn't contain openpi. Add openpi and remove definition here.
class ExploreNoiseNet(nn.Module):
    """
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation_type: str,
        noise_logvar_range: list,  # [min_std, max_std]
        noise_scheduler_type: str,
    ):
        super().__init__()
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims + [out_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
        )
        self.noise_scheduler_type = noise_scheduler_type
        self.set_noise_range(noise_logvar_range)

    def set_noise_range(self, noise_logvar_range: list):
        self.noise_logvar_range = noise_logvar_range
        noise_logvar_min = self.noise_logvar_range[0]
        noise_logvar_max = self.noise_logvar_range[1]
        self.register_buffer(
            "logvar_min",
            torch.log(torch.tensor(noise_logvar_min**2, dtype=torch.float32)).unsqueeze(
                0
            ),
        )
        self.register_buffer(
            "logvar_max",
            torch.log(torch.tensor(noise_logvar_max**2, dtype=torch.float32)).unsqueeze(
                0
            ),
        )

    def forward(self, noise_feature: torch.Tensor):
        if "const" in self.noise_scheduler_type:  # const or const_schedule_itr
            # pick the lowest noise level when we use constant noise schedulers.
            noise_std = torch.exp(0.5 * self.logvar_min)
        else:
            # use learnable noise level.
            noise_logvar = self.mlp_logvar(noise_feature)
            noise_std = self.post_process(noise_logvar)
        return noise_std

    def post_process(self, noise_logvar):
        """
        input:
            torch.Tensor([B, Ta , Da])
        output:
            torch.Tensor([B, Ta, Da])
        """
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = (
            self.logvar_min
            + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        )
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


activation_dict = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
    }
)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="tanh",
        out_activation_type="identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))  # type: ignore
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))  # type: ignore

            # Add activation function
            act = (
                activation_dict[activation_type.lower()]
                if idx != num_layer - 1
                else activation_dict[out_activation_type.lower()]
            )
            layers.append(("act_1", act))  # type: ignore

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][
                0
            ]  # Linear layer is first in the last Sequential # type: ignore
            nn.init.constant_(final_linear.bias, out_bias_init)
            logger.info(
                f"Initialized the bias of the final linear layer to {out_bias_init}"
            )

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class FlowMatchingActionHeadForRLActionPrediction(FlowmatchingActionHead):
    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__(config)

        # TODO(lx): move to config.
        self.fm_config_joint_logprob = False
        self.fm_config_noise_method = "flow_sde"
        self.fm_config_ignore_last = False
        self.fm_config_safe_get_logprob = False
        self.fm_config_noise_anneal = False
        self.fm_config_noise_params = [0.7, 0.3, 400]
        self.fm_config_noise_level = 0.5
        self.fm_config_add_value_head = False
        self.temp_valid_action_dim = (
            7  # TODO(lx): This is infered from metadata, implement in future.
        )

        self.value_head = ValueHead(
            input_dim=self.input_embedding_dim,
            hidden_sizes=(512, 256, 128),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

        if self.fm_config_noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.input_embedding_dim,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    # TODO(lx): pi0 implementation says that this function contains potential nan, need to check.
    # the safe_get_logprob is changed to torch implementation
    def get_logprob_norm(self, sample, mu, sigma):
        if self.fm_config_safe_get_logprob:
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        Pay attention: The time notation of gr00t is different from openpi.
        In gr00t, the time is from 0 to 1, while in openpi, the time is from 1 to 0.
        """
        # expand the shape
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.fm_config_noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.fm_config_noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.fm_config_noise_level).to(device)

        # velocity prediction
        t_cont = idx[0] / float(denoise_steps)
        # TODO(lx): In training, will idx be a tensor contains different values?
        t_discretized = int(t_cont * self.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(bsize,), fill_value=t_discretized, device=device
        )
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        # value prediction
        if self.fm_config_add_value_head and compute_values:
            # TODO(lx): In current setting, Gr00t has no action chunk property. It make no difference using chunk_critic_input.
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    model_output[:, -self.action_horizon :], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(model_output, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)

        # ode/sde sampling
        pred = self.action_decoder(model_output, embodiment_id)
        v_t = pred[:, -self.action_horizon :]

        # TODO(lx): A little messy here. For Now we keep the same code structure as openpi.
        timesteps = torch.linspace(0, 1, denoise_steps + 1, device=device)
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        # Emphasize: In Gr00t, x0: noise, x1: data.
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = (
                t_input + delta
            )  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.fm_config_noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                # https://zhuanlan.zhihu.com/p/1961533469726335106
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.fm_config_noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
                x1_weight = t_input + delta
                x_t_std = (1 - (t_input + delta)) * sin_term
            elif self.fm_config_noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = self.reinflow_explore_noise_net(model_output)
            else:
                raise ValueError(f"Invalid noise method: {self.fm_config_noise_method}")
        # In eval, this equals to x_t_mean = x_t + v*dt(dt>0).
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ) -> BatchFeature:
        # TODO(lx) This function will finally replace get_action() as long as we prove its correctness.

        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []
        values = []
        if self.fm_config_joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        # determine the denoise step for the logprob calculation
        if mode == "train":
            if self.fm_config_joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.fm_config_noise_method == "flow_sde":
                    if self.fm_config_ignore_last:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
                elif self.fm_config_noise_method == "flow_cps":
                    # the last denoising step of the flow-cps is deterministic
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
                elif self.fm_config_noise_method == "reinflow":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Run denoising steps.
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )
            else:
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, -self.config.action_horizon :, : self.temp_valid_action_dim
        ]
        values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return BatchFeature(
            data={"action_pred": x_0}
        ), {  # this is for gr00t validity check
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def get_log_prob_value(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]

        chains_log_probs = []
        chains_values = []
        if self.fm_config_joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=idx,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=num_steps,
                compute_values=compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)
            chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )


class GR00T_N1_5_ForRLActionPrediction(GR00T_N1_5):
    """
    GR00T_N1_5 model for reinforcement learning action prediction.
    It's a combination of the Gr00tPolicy and GR00T_N1_5 model.

    Notes:
        - Device is handled by huggingface worker.
        - EmbodimentTag determines the state encoder and action head to use.
          we use "new_embodiment" reserved by gr00t.

    """

    _no_split_modules = [
        "Eagle2_5_VLForConditionalGeneration",
        "FlowMatchingActionHeadForRLActionPrediction",
        "TimestepEncoder",
        "TimestepEmbedding",
        "ValueHead",
    ]

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
    ):
        super().__init__(config, local_model_path)
        # The param loading is after construction in from_pretrained(), so it should be safe to to so.
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowMatchingActionHeadForRLActionPrediction(action_head_cfg)

        self._modality_config = modality_config  # ModalityConfig(delta_indices=[0], modality_keys=['video.ego_view'])
        self._modality_transform = modality_transform
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        if denoising_steps is not None:
            if hasattr(self, "action_head") and hasattr(
                self.action_head, "num_inference_timesteps"
            ):
                self.action_head.num_inference_timesteps = denoising_steps

        # TODO(lx): meta_data are from training, when embodiment_tag is new, no metadata available.
        # so we borrow tag from "gr1"
        self._load_metadata(self.model_path / "experiment_cfg")

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    def convert_obs_to_gr00t_format(self, env_obs):
        """
        Convert the observation to the format expected by the GR00T model.
        The data format is determined by the modality_config and meta/info.json following LeRobot format.
        Considering that we don't have a unified data inferface, we use direct logic here.
        """
        groot_obs = {}
        # video
        # TODO(lx): If we use new embodiment tag, this can be avoided. But now we have to resize images to GR1 data version.
        env_obs["images"] = cut_and_resize_images(
            env_obs["images"], env_obs["images"].shape[-2], 256
        )
        # [B, C, H, W] -> [B, T(1), C, H, W] -> [B, T, H, W, C]
        images = env_obs["images"].unsqueeze(1).numpy()
        groot_obs["video.ego_view"] = np.transpose(images, (0, 1, 3, 4, 2))
        # state
        if "state" in env_obs:
            raise NotImplementedError("State from simulation are not unified yet.")
        else:
            # gr00t pad the state to input dimension
            #  create state of [B, T, C]
            groot_obs["state.left_arm"] = np.zeros((env_obs["images"].shape[0], 1, 7))
        # annotation
        groot_obs["annotation.human.action.task_description"] = env_obs[
            "task_descriptions"
        ]
        return groot_obs

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def forward(
        self,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
    ) -> Dict[str, Any]:
        normalized_input = {
            "state": data["state"],
            "state_mask": data["state_mask"],
            "eagle_input_ids": data["eagle_input_ids"],
            "eagle_attention_mask": data["eagle_attention_mask"],
            "eagle_pixel_values": data["eagle_pixel_values"],
            "eagle_image_sizes": data["eagle_image_sizes"],
            "embodiment_id": data["embodiment_id"],
        }
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        log_probs, value_t = self.action_head.get_log_prob_value(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            -self.action_head.action_horizon :,
            : self.action_head.temp_valid_action_dim,
        ]
        # post process
        if self.action_head.fm_config_joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = data["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = data["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_horizon,
                : self.action_head.temp_valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "prev_logprobs": prev_logprobs,
            "values": value_t,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        """
        TODO(lx): remove the mode flag in future. Now we set mode to facilitate future seperate validation.
        """
        observations = self.convert_obs_to_gr00t_format(env_obs)
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)

        if mode == "eval":  # mid test
            normalized_action, result = self._get_rl_action(normalized_input)
            unnormalized_action = self._get_unnormalized_action(normalized_action)

            if not is_batch:
                unnormalized_action = squeeze_dict_values(unnormalized_action)

            # Accord to gr1 definition, action.left_arm happens to be 7 dims, matching the demand of maniskill.
            # TODO(lx): maniskill_env.py line 254 shows that all the action chunk are used for env forward. It's reasonable?
            raw_action = unnormalized_action["action.left_arm"]

            return raw_action, result
        else:  # mode == "train", the code are not from gr00t. We test the code with RL training and compare the performance.
            normalized_action, result = self._get_rl_action(normalized_input)
            unnormalized_action = self._get_unnormalized_action(normalized_action)

            if not is_batch:
                unnormalized_action = squeeze_dict_values(unnormalized_action)

            # Accord to gr1 definition, action.left_arm happens to be 7 dims, matching the demand of maniskill.
            # TODO(lx): maniskill_env.py line 254 shows that all the action chunk are used for env forward. It's reasonable?
            raw_action = unnormalized_action["action.left_arm"]

            return raw_action, result

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def _get_rl_action(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.compute_dtype
        ):
            # We expand get_action() and replace action head inference with RL inference.
            backbone_inputs, action_inputs = self.prepare_input(normalized_input)
            # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
            backbone_outputs = self.backbone(backbone_inputs)
            action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
                backbone_outputs, action_inputs
            )
            actions = rlinf_outputs["actions"]
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
            actions = actions.float()

        inputs_data = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "inputs_data": inputs_data,
        }

        return actions, result

    def _get_action_from_normalized_input(
        self, normalized_input: Dict[str, Any]
    ) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.compute_dtype
        ):
            model_pred = self.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self, normalized_action: torch.Tensor
    ) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata


#######################################################################################################
def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """
    Cut and resize the images to the crop size.
    """
    original_width = images.shape[-1]  # 640
    start = (original_width - crop_size) // 2  # (640-480)/2 = 80
    end = start + crop_size  # 80 + 480 = 560

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images[:, :, :, start:end]  # Shape: (2, 3, 480, 480)
    # Step 2: Resize the cropped 480x480 tensor to 256x256
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )
    return resized_tensor


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
