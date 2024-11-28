# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from functools import partial
from typing import Callable, Union

import torch

from ..adalora.gram_schmidt import gram_schmidt_orthonormalize_model
from .config import IncreLoraConfig
from .layer import SVDLinear
from .model import IncreLoraModel


logger = logging.getLogger(__name__)


class RankAllocator:
    """
    The RankAllocator for IncreLoraModel. Paper: https://arxiv.org/abs/2308.12043

    Args:
        config ([`IncreLoraConfig`]): The configuration of the IncreLora model.
        model: the model that we apply IncreLoRA to.

    """

    total_current_rank: int
    total_modules: int
    total_steps: int

    def __init__(
        self,
        model: IncreLoraModel,
        peft_config: IncreLoraConfig,
        adapter_name: str,
        track_metrics: Callable = lambda _: None,
    ):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.alternative_scoring: bool = self.peft_config.alternative_scoring
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        self.track_metrics = track_metrics
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()

    @property
    def total_target_rank(self) -> int:
        return self.peft_config.target_r * self.total_modules

    @property
    def top_h(self):
        return self.peft_config.top_h

    @property
    def reserve_ranks(self):
        return self.peft_config.reserve_ranks

    def setup(self, *, total_steps, optimizer, model, weight_decay):
        total_modules: int = 0
        for layer in model.modules():
            if isinstance(layer, SVDLinear):
                total_modules += 1

        self.total_steps = total_steps
        self.total_modules = total_modules
        self.total_current_rank = total_modules * self.peft_config.init_r

        self.weight_decay = weight_decay

        rank_per_round = self.top_h * self.reserve_ranks

        total_additional_rank = self.total_modules * (self.peft_config.target_r - self.peft_config.init_r)

        num_rounds = math.ceil(total_additional_rank / rank_per_round)

        total_incre_step = self.peft_config.deltaT * num_rounds

        logger.info(
            "Total incremental step: total_incre_step: %d, of total steps: %.2f (%d modules)",
            total_incre_step,
            total_incre_step / total_steps,
            self.total_modules,
        )

        new_params = []

        for module in model.modules():
            if isinstance(module, SVDLinear):
                new_params.extend(module.add_reserve_ranks(self.adapter_name, self.peft_config.reserve_ranks))

        optimizer.add_param_group(
            {
                "params": new_params,
                "weight_decay": self.weight_decay,
            }
        )

        self.peft_config.rank_pattern = model.get_rank_pattern(self.adapter_name)

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, layer in model.named_modules():
            if not isinstance(layer, SVDLinear):
                continue

            if n not in self.ipt:
                if self.alternative_scoring:
                    zeros = partial(
                        torch.zeros,
                        len(layer.lora_E),
                        requires_grad=False,
                    )

                    self.ipt[n] = zeros()
                    self.exp_avg_ipt[n] = zeros()
                    self.exp_avg_unc[n] = zeros()

                else:
                    self.ipt[n] = 0
                    self.exp_avg_ipt[n] = 0
                    self.exp_avg_unc[n] = 0

            self.ipt[n] = layer.score

            # Update sensitivity
            self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
            # Update uncertainty
            self.exp_avg_unc[n] = (
                self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
            )

    def retrieve_scores(self, model) -> dict[str, torch.Tensor]:
        module_scores: dict[str, torch.Tensor] = {}

        # Calculate the importance score for each sub matrix
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                if self.alternative_scoring:
                    module_scores[n] = (
                        self.exp_avg_ipt[n][-self.reserve_ranks:] * self.exp_avg_unc[n][-self.reserve_ranks:]
                    )
                else:
                    module_scores[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n]

        return module_scores

    def increase_layer_rank(self, layer: SVDLinear, ranks_to_add: Union[int, list[bool]]) -> list[torch.nn.Parameter]:
        """Add the selected ranks to the layer.

        Args:
            ranks_to_add: specifies either the number of ranks to add
            (i.e. all reserve ranks from the back of the parameter list)
            or a boolean mask specifying which ranks to add.

        Returns:
            The parameters that need to be added to the optimizer.
        """

        num_added: int
        if isinstance(ranks_to_add, int):
            num_added = ranks_to_add
            ranks_to_add = [True] * ranks_to_add
        else:
            num_added = sum(ranks_to_add)

        layer.r[self.adapter_name] += num_added
        self.total_current_rank += num_added

        new_paramters: list[torch.nn.Parameter] = []

        # Make the existing lora_E parameters trainable
        for i, (add, param_e) in enumerate(zip(ranks_to_add, layer.lora_E[self.adapter_name])):
            if not add:
                continue
            # Param already existed, but wasn't trained before
            param_e.requires_grad = True
            new_paramters.append(param_e)
            layer.rank_pattern[self.adapter_name][i] = True

        new_paramters.extend(layer.add_reserve_ranks(self.adapter_name, num_added))
        return new_paramters

    def increase_to_target_rank(self, model, optimizer):
        module_scores = self.retrieve_scores(model)

        metrics = {}

        # Calculate the increasing threshold
        if self.alternative_scoring:
            k = min(self.top_h * self.reserve_ranks, self.total_target_rank - self.total_current_rank)
        else:
            k = min(self.top_h, self.total_target_rank - self.total_current_rank)

        all_scores = torch.cat(list(module_scores.values()))

        increase_threshold = torch.topk(all_scores, k)[0][-1].item()

        with torch.no_grad():
            new_param_list = []
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    num_added: int = 0
                    if self.alternative_scoring:
                        ranks_to_add = (module_scores[n] >= increase_threshold).tolist()

                        if any(ranks_to_add):
                            num_prev_params = len(layer.lora_E[self.adapter_name]) - len(ranks_to_add)

                            # fill up the mask
                            ranks_to_add = [False] * num_prev_params + ranks_to_add

                            num_added = sum(ranks_to_add)

                    elif module_scores[n] >= increase_threshold:
                        ranks_to_add = self.peft_config.reserve_ranks
                        num_added = ranks_to_add

                    if num_added > 0:
                        self.increase_layer_rank(layer, ranks_to_add)
                        self.peft_config.rank_pattern[n] = layer.rank_pattern
                        logger.info("The lora parameters rank of %s increased by %d", n, num_added)

                    # log metrics
                    metrics[f"num_rank/{n}"] = layer.r[self.adapter_name]

            metrics["total_rank"] = self.total_current_rank

            optimizer.add_param_group(
                {
                    "params": new_param_list,
                    "weight_decay": self.weight_decay,
                }
            )

            if self.total_current_rank == self.total_target_rank:
                for module in model.modules():
                    if isinstance(module, SVDLinear):
                        for param in module.lora_E[self.adapter_name][-self.peft_config.reserve_ranks:]:
                            param.fill_(0.0)

            metrics["budget/total_rank"] = self.total_current_rank
            metrics["budget/increase_threshold"] = increase_threshold

            self.track_metrics(metrics)

        return increase_threshold

    def update_and_allocate(self, model, global_step, optimizer, training_args, **kw):
        if self.total_current_rank < self.total_target_rank:
            self.update_ipt(model)
            if (
                global_step > training_args.get_warmup_steps(self.total_steps)  # warmup complete
                and global_step % self.peft_config.deltaT == 0  # at growth step
            ):
                self.increase_to_target_rank(model, optimizer)

        if self.peft_config.orthonormalize:
            gram_schmidt_orthonormalize_model(model)

        metrics = {}

        def compute_and_log(mat_cov, name):
            I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
            I.requires_grad = False
            orth_regu = torch.norm(mat_cov - I, p="fro")
            regu_loss.append(orth_regu.item())
            metrics[f"Orth_regu_loss/{name}"] = orth_regu.item()

        if global_step % 250 == 0:
            with torch.no_grad():
                regu_loss = []
                for n, layer in model.named_modules():
                    if isinstance(layer, SVDLinear):
                        wA = torch.cat(list(layer.lora_A[self.adapter_name]), 0)
                        wB = torch.cat(list(layer.lora_B[self.adapter_name]), 1)
                        mat_cov_A = wA @ wA.T
                        mat_cov_B = wB.T @ wB
                        compute_and_log(mat_cov_A, n + ".lora_A")
                        compute_and_log(mat_cov_B, n + ".lora_B")

                metrics["train/orth_regu_loss"] = sum(regu_loss) / len(regu_loss)

            self.track_metrics(metrics)
