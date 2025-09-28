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

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.model_utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)
from rlinf.models.embodiment.modules.value_head import ValueHead


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction):
    def __init__(
        self, config: OpenVLAOFTConfig, action_dim, num_action_chunks, add_value_head
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        )

        if self.config.vh_mode is not None and add_value_head:
            self.hidden_size = self.config.hidden_size
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(self.hidden_size, output_dim=output_dim)

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm label & mask & embedding
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )

        input_embeddings = self.get_input_embeddings()(input_ids)  # [B, L + act + 1, D]
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        # vision
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        # [B, 256 * num_images, D]
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        # multimodal embeddings
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        assert (
            multimodal_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            multimodal_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.action_dim * self.num_action_chunks))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = mask * repeat_factor

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        return actions

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        env_obs=None,
        calulate_logprobs=True,
        calulate_values=True,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")
        processed_obs = None
        if env_obs is not None:
            assert env_obs is not None
            task_descriptions = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            image_tensor = env_obs["images"]
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(1)
            assert image_tensor.ndim == 5

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype
            images = {"images": image_tensor}
            processed_obs = self.input_processor(
                text=task_descriptions,
                images=images,
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=max_length,
            )

            input_ids = processed_obs["input_ids"].to(device=device, dtype=precision)
            attention_mask = processed_obs["attention_mask"].to(
                device=device, dtype=precision
            )
            pixel_values = processed_obs["pixel_values"].to(
                device=device, dtype=precision
            )

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        n_prompt_tokens = input_ids.shape[-1] - 1
        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            attention_mask[:, -1 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = outputs.hidden_states[-1]  # (B, seq_len, D)
        assert last_hidden_states.shape[1] == mm_embeddings.shape[1]

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * self.num_action_chunks,
            :,
        ]  # [B, act, vocab_size + 64]

        last_hidden_states = last_hidden_states[
            :, -self.action_dim * self.num_action_chunks - 1 : -1
        ]

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        processed_logits_tensor = logits_tensor / kwargs.get("temperature", 1.0)
        top_k = min(
            kwargs.get("top_k", 50), processed_logits_tensor.size(-1)
        )  # Safety check
        if top_k > 0:
            logits_warper = TopKLogitsWarper(
                top_k
            )  # since here is logprob instead of logits, we use 0 instead of -inf
            processed_logits_tensor = logits_warper(None, processed_logits_tensor)

        processed_logprob_tensor = F.log_softmax(
            processed_logits_tensor, dim=-1
        )  # [B, act, vocab_size + 64]

        if do_sample:
            probs_tensor = torch.exp(
                processed_logprob_tensor
            )  # [B, act, vocab_size + 64]
            probs_flat = probs_tensor.view(
                -1, processed_logprob_tensor.shape[-1]
            )  # [B * act, vocab_size + 64]

            sample_flat = torch.multinomial(
                probs_flat, num_samples=1, replacement=True
            )  # [B * act, 1]
            idxs = sample_flat.view(
                processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
            )  # [B, act]
        else:
            idxs = processed_logprob_tensor.argmax(dim=-1)  # [B, act]

        # assert torch.all(idxs >= 0) and torch.all(idxs < self.config.n_action_bins)
        # generated_ids = idxs + (self.vocab_size - self.config.n_action_bins)
        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        chunk_action_tokens = idxs.reshape(-1, self.action_dim)
        predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        # normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )  # [B, dim]
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor.permute(
            0, 2, 1
        )  # [B, vocab-size, action-dim]
        action_logits[:, : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[:, self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

        if hasattr(self, "value_head") and calulate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        chunk_action_tokens = idxs.reshape(-1, self.num_action_chunks, self.action_dim)

        inputs_data = processed_obs
        inputs_data["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "inputs_data": inputs_data,
        }

        return chunk_actions, result

    def preprocess_for_train(self, data):
        # action-token: [bsz, chunk-step, action-dim] -> [bsz, chunk-step x action-dim]
        for key in ["action_tokens"]:
            value = data[key]
            data[key] = value.reshape(
                value.shape[0],
                self.action_dim * self.num_action_chunks,
                *value.shape[3:],
            )
        return data

    def setup_config_and_processor(self, model_config, cfg, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0
        self.policy_setup = cfg.actor.model.policy_setup
        self.max_prompt_length = cfg.runner.max_prompt_length

        self.input_processor = input_processor

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        output_hidden_states: bool = False,
        data: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
    ):
        if data is not None:
            data = self.preprocess_for_train(data)
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            pixel_values = data["pixel_values"]

            action_tokens = data["action_tokens"]

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            input_ids[:, -self.action_dim * self.num_action_chunks - 2] == 29871
        )
        assert torch.all(
            attention_mask[:, -2 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        if compute_values:
            output_hidden_states = True

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, :, -self.action_dim * self.num_action_chunks : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / data.get("temperature", 1.0)
            top_k = min(
                data.get("top_k", -1), processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor.permute(
                0, 2, 1
            )  # [B, vocab-size, action-dim]
            action_logits[:, : self.vocab_size - self.config.n_action_bins] = -torch.inf
            action_logits[:, self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result
