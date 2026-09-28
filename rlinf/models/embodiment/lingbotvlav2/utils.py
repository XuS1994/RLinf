# Copyright 2026 The RLinf Authors.
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

"""Flow-SDE transitions and trajectory densities for LingBot-VLA 2.0."""

import torch


def flow_sde_transition(
    x: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 Gaussian transition used by sampling and PPO replay.

    At t=1 the diffusion denominator uses the next time, as in RLinf's
    LingBot-VLA/OpenPI flow-SDE. Every step has positive variance, including
    the final step. Scores are densities of the complete latent path.
    """
    x, velocity = x.float(), velocity.float()
    denominator = torch.where(time == 1, delta, 1 - time)
    sigma_squared = noise_level**2 * time / denominator
    mean = (
        x
        - delta * velocity
        - sigma_squared * delta / (2 * time) * (x + (1 - time) * velocity)
    )
    std = torch.sqrt(delta * sigma_squared)
    return mean, std


def gaussian_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Score a detached continuous sample without reducing latent dimensions."""
    return torch.distributions.Normal(mean.float(), std.float()).log_prob(
        sample.float()
    )
