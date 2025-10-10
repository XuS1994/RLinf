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
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config


class GR00T_N1_5_ForRLActionPrediction(GR00T_N1_5):
    """
    GR00T_N1_5 model for reinforcement learning action prediction.
    It's a combination of the Gr00tPolicy and GR00T_N1_5 model.

    Notes:
        - Device is handled by huggingface worker.
        - EmbodimentTag determines the state encoder and action head to use.
          we use "new_embodiment" reserved by gr00t.

    """

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
        self._modality_config = modality_config  # ModalityConfig(delta_indices=[0], modality_keys=['video.ego_view'])
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode TODO(lx): fix this in finetuning mode.
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
                print(f"Set action denoising steps to {denoising_steps}")

        # TODO(lx): meta_data are from training, when embodiment_tag is new, no metadata available.
        # so we borrow tag from "gr1"
        self._load_metadata(self.model_path / "experiment_cfg")

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

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs=None,
        **kwargs,
    ):
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
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        # Accord to gr1 definition, action.left_arm happens to be 7 dims, matching the demand of maniskill.
        # TODO(lx): maniskill_env.py line 254 shows that all the action chunk are used for env forward. It's wrong, need to fix it.
        raw_action = unnormalized_action["action.left_arm"]

        return raw_action, {}

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
