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
from typing import Callable, Optional

import torch

from utils.orthonormalization import orthonormalize_model, normalize_model
from .config import GrowRAConfig
from .layer import SVDLinear
from .model import GrowRAModel

from peft.utils.other import get_pattern_key

logger = logging.getLogger(__name__)


class RankAllocator:
    """
    The RankAllocator for GrowRAModel. Paper: https://arxiv.org/abs/2308.12043

    Args:
        config ([`GrowRAConfig`]): The configuration of the GrowRA model.
        model: the model that we apply IncreLoRA to.

    """

    total_current_rank: int
    total_modules: int
    total_steps: int

    def __init__(
        self,
        model: GrowRAModel,
        peft_config: GrowRAConfig,
        adapter_name: str,
        track_metrics: Callable = lambda _: None,
        random_selection: bool = False,
        ignore_uncertainty: bool = False,
        target_rank_pattern: Optional[dict[str, int]] = None
    ):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        self.track_metrics = track_metrics
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.random_selection = random_selection
        self.ignore_uncertainty = ignore_uncertainty

        self.target_rank_pattern = target_rank_pattern

        if random_selection:
            logger.warning("Using random rank selection.")

        self.reset_ipt()

    @property
    def total_target_rank(self) -> int:
        return self.peft_config.target_r * self.total_modules

    @property
    def top_h(self):
        return self.peft_config.num_top_modules

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

        model.growing = True

        logger.info("Total steps: %d; total modules: %d", self.total_steps, self.total_modules)


        self.weight_decay = weight_decay

        rank_per_round = self.top_h * self.reserve_ranks

        total_additional_rank = self.total_modules * (self.peft_config.target_r - self.peft_config.init_r)

        logger.info("Initial rank: %d; total additional ranks: %d", self.peft_config.init_r, total_additional_rank)


        num_rounds = math.ceil(total_additional_rank / rank_per_round)

        total_incre_step = self.peft_config.growth_interval * num_rounds

        logger.info("Growing %d ranks per round => %d rounds of growth", rank_per_round, num_rounds)

        logger.info(
            "Total growth phase steps: %d; fraction of total steps: %.2f",
            total_incre_step,
            total_incre_step / total_steps,
        )

        new_params = model.setup_reserve_ranks()

        if len(new_params) > 0:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "weight_decay": self.weight_decay,
                }
            )

        self.peft_config.rank_pattern = model.get_rank_pattern(self.adapter_name)

    def reset_ipt(self):
        if self.peft_config.reserve_rank_scoring:
            self.exp_avg_grad = {}
        else:
            self.exp_avg_ipt = {}

        self.exp_avg_unc = {}

    @staticmethod
    def moving_avg(avg, current, *, beta):
        if avg is None:
            return current

        avg = (
            beta * avg + (1 - beta) * current[:avg.size(0)]
        )

        if avg.size(0) < current.size(0):
            avg = torch.cat([avg, current[avg.size(0):]])

        return avg

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, layer in model.named_modules():
            if not isinstance(layer, SVDLinear):
                continue

            if self.peft_config.reserve_rank_scoring:
                self.exp_avg_grad[n] = self.moving_avg(
                    self.exp_avg_grad.get(n, None),
                    layer.e_grad,
                    beta=self.beta1
                )

                unc = (layer.e_grad - self.exp_avg_grad[n]).abs()

                # Reset the gradient
                layer.e_grad = None

            else:
                self.exp_avg_ipt[n] = self.moving_avg(
                    self.exp_avg_ipt.get(n, None),
                    layer.ipt,
                    beta=self.beta1
                )

                unc = (layer.ipt - self.exp_avg_ipt[n]).abs()

            self.exp_avg_unc[n] = self.moving_avg(self.exp_avg_unc.get(n, None), unc, beta=self.beta2)

    def retrieve_scores(self, model) -> dict[str, torch.Tensor]:
        module_scores: dict[str, torch.Tensor] = {}

        # Calculate the importance score for each sub matrix
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                reserve = layer.get_reserve_mask(self.adapter_name)

                dtype = layer.lora_E[self.adapter_name][0].dtype
                device = layer.lora_E[self.adapter_name][0].device
                num_reserve = len(reserve)

                if self.random_selection:
                    module_scores[n] = torch.rand(num_reserve, device=device, dtype=dtype)

                elif self.target_rank_pattern is not None:
                    target_rank_key = get_pattern_key(self.target_rank_pattern, n)
                    target_rank = self.target_rank_pattern.get(target_rank_key, 0)

                    current_rank = layer.r[self.adapter_name]
                    reserve_ranks = self.reserve_ranks
                    remaining_ranks = target_rank - current_rank

                    module_scores[n] = torch.full((num_reserve, ), -1, device=device, dtype=dtype)

                    if remaining_ranks > 0:
                        mod = remaining_ranks % reserve_ranks
                        div = remaining_ranks // reserve_ranks

                        module_scores[n][:mod] = div + 1

                        if div > 0:
                            module_scores[n][mod:] = div

                        module_scores[n] += 0.5 * torch.rand(num_reserve, device=device, dtype=dtype)

                elif self.ignore_uncertainty:
                    module_scores[n] = self.exp_avg_grad[n][reserve].abs()

                else:
                    module_scores[n] = self.exp_avg_grad[n][reserve].abs() * self.exp_avg_unc[n][reserve]

        return module_scores

    def increase_layer_rank(self, layer: SVDLinear, ranks_to_add: list[bool]) -> list[torch.nn.Parameter]:
        """Add the selected ranks to the layer.

        Args:
            ranks_to_add: specifies either the number of ranks to add
            (i.e. all reserve ranks from the back of the parameter list)
            or a boolean mask specifying which ranks to add.

        Returns:
            The parameters that need to be added to the optimizer.
        """
        lora_E = layer.lora_E[self.adapter_name]

        num_added: int = sum(ranks_to_add)

        layer.r[self.adapter_name] += num_added
        self.total_current_rank += num_added

        new_paramters: list[torch.nn.Parameter] = []

        assert len(ranks_to_add) == len(lora_E)
        assert len(layer.rank_pattern[self.adapter_name]) == len(lora_E)

        # print(layer.rank_pattern[self.adapter_name])
        # print(ranks_to_add)
        # print([p.requires_grad for p in lora_E])

        # Make the existing lora_E parameters trainable
        for i, (add, param_e) in enumerate(zip(ranks_to_add, lora_E)):
            if not add:
                continue
            # Param already existed, but wasn't trained before
            assert not layer.rank_pattern[self.adapter_name][i]
            assert not param_e.requires_grad

            param_e.requires_grad = True
            new_paramters.append(param_e)

            if not self.peft_config.advance_learn:
                layer.lora_A[self.adapter_name][i].requires_grad = True
                layer.lora_B[self.adapter_name][i].requires_grad = True

                new_paramters.append(layer.lora_A[self.adapter_name][i])
                new_paramters.append(layer.lora_B[self.adapter_name][i])

            layer.rank_pattern[self.adapter_name][i] = True

        new_paramters.extend(layer.add_reserve_ranks(self.adapter_name, num_added))
        return new_paramters

    def increase_to_target_rank(self, model, optimizer, params_groups_kws):
        module_scores = self.retrieve_scores(model)

        metrics = {}

        # Calculate the increasing threshold
        k = min(self.top_h * self.reserve_ranks, self.total_target_rank - self.total_current_rank)

        if not k > 0:
            return float("Inf")

        all_scores = torch.cat(list(module_scores.values()))

        values, _ = torch.topk(all_scores, k)
        increase_threshold = values[-1].item()

        with torch.no_grad():
            new_param_list = []
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    ranks_to_add: list[bool]

                    # one booelan per reserve rank
                    ranks_to_add = (module_scores[n] >= increase_threshold).tolist()

                    if any(ranks_to_add):
                        # map the reserve rank flags to flags for the complete list of params
                        add_rank = iter(ranks_to_add)

                        # next(add_rank) is only called if the current rank is a reserve rank
                        # otherwise, the iterator is not progressed (as we only have ranks_to_add
                        # values for the reserve ranks)
                        ranks_to_add = [(not p) and next(add_rank) for p in layer.rank_pattern[self.adapter_name]]

                        new_param_list.extend(self.increase_layer_rank(layer, ranks_to_add))

                        self.peft_config.rank_pattern[n] = layer.rank_pattern
                        logger.info("The lora parameters rank of %s increased by %d", n, sum(ranks_to_add))

                    # log metrics
                    metrics[f"num_rank/{n}"] = layer.r[self.adapter_name]

            if len(new_param_list) > 0:
                optimizer.add_param_group(
                    {
                        "params": new_param_list,
                        **params_groups_kws
                    }
                )

            if self.total_current_rank >= self.total_target_rank:
                model.drop_reserve()
                self.growing = False

            metrics["budget/total_rank"] = self.total_current_rank
            metrics["budget/avg_rank"] = self.total_current_rank / self.total_modules
            metrics["budget/increase_threshold"] = increase_threshold

            self.track_metrics(metrics)

        return increase_threshold

    def update_and_allocate(self, model, global_step, optimizer, training_args, **kw):
        if self.total_current_rank < self.total_target_rank:
            self.update_ipt(model)

            growth_delay = self.peft_config.growth_delay

            if growth_delay is None:
                growth_delay = training_args.get_warmup_steps(self.total_steps)


            if (global_step == -1) or (global_step >= 0 and (1 + global_step - growth_delay) % self.peft_config.growth_interval == 0):
                remaining_steps = self.total_steps - global_step

                self.increase_to_target_rank(
                    model, optimizer,
                    params_groups_kws={
                        "weight_decay": self.weight_decay,
                        "initial_step": global_step,
                        "warmup_steps": training_args.get_warmup_steps(remaining_steps),
                        "remaining_steps": remaining_steps,
                    }
                )

        if model.growing or not self.peft_config.disable_orthnorm_after_growth_complete:
            if self.peft_config.orthonormalize:
                orthonormalize_model(
                    model,
                    reserve_only=self.peft_config.orthonormalize_reserve_only,
                    ignore_non_reserve=self.peft_config.orthonormalize_ignore_non_reserve,
                    normalize=self.peft_config.normalize,
                )
            elif self.peft_config.normalize:
                normalize_model(model, reserve_only=self.peft_config.normalize_reserve_only)

        if global_step % training_args.logging_steps == 0:
            metrics = {}

            def compute_and_log(mat_cov, name):
                global orthogonal_loss_sum, normalization_loss_sum, num_elements

                I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                I.requires_grad = False

                m = mat_cov - I

                m_sqrt = m*m

                norm_loss = torch.sqrt(torch.trace(m_sqrt)).item()

                diag_indices = torch.arange(m_sqrt.shape[0])

                m_sqrt[diag_indices, diag_indices] = 0

                orth_loss = torch.sqrt(torch.sum(m_sqrt)).item()

                #if global_step % 100 == 0:
                #    metrics[f"Norm_loss/{name}"] = norm_loss
                #    metrics[f"Orth_loss/{name}"] = orth_loss

                return norm_loss, orth_loss

            with torch.no_grad():
                orthogonal_loss_sum = 0
                normalization_loss_sum = 0
                num_elements = 0

                for n, layer in model.named_modules():
                    if isinstance(layer, SVDLinear):
                        wA = torch.cat(list(layer.lora_A[self.adapter_name]), 0)
                        wB = torch.cat(list(layer.lora_B[self.adapter_name]), 1)
                        mat_cov_A = wA @ wA.T
                        mat_cov_B = wB.T @ wB
                        nla, ola = compute_and_log(mat_cov_A, n + ".lora_A")
                        nlb, olb = compute_and_log(mat_cov_B, n + ".lora_B")

                        orthogonal_loss_sum += ola + olb
                        normalization_loss_sum += nla + nlb

                        num_elements += 2

                metrics["loss_normal"] = normalization_loss_sum / num_elements
                metrics["loss_orthogonal"] = orthogonal_loss_sum / num_elements

            self.track_metrics(metrics)
