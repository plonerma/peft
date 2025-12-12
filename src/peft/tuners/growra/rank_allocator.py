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

from typing import Callable, Iterable, Optional
import logging
import math
from collections import defaultdict

import torch
from peft.utils.other import get_pattern_key

from utils.orthonormalization import orthonormalize_model, normalize_model
from .config import GrowRAConfig
from .layer import SVDLinear
from .model import GrowRAModel



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
        target_rank_pattern: Optional[dict[str, int]] = None,
        reserve_weight_decay: Optional[float] = None,
        reserve_separate_groups: bool = False,
        reserve_constant_lr: bool = False,
        reserve_lr: Optional[float] = None,
        reserve_betas: Optional[tuple[float, float]] = None,
        reinit_after_growth: bool = False,

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

        self.reserve_weight_decay = reserve_weight_decay
        self.reserve_constant_lr = reserve_constant_lr
        self.reserve_betas = reserve_betas
        self.reserve_lr = reserve_lr
        self.reserve_separate_groups = reserve_separate_groups
        self.reinit_after_growth = reinit_after_growth

        if random_selection:
            logger.warning("Using random rank selection.")

        self.new_main_params = []
        self.new_reserve_params = []
        self.old_reserve_params = []

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
        if self.reserve_weight_decay is None:
            self.reserve_weight_decay = weight_decay

        rank_per_round = self.top_h * self.reserve_ranks

        total_additional_rank = self.total_modules * (self.peft_config.target_r - self.peft_config.init_r)

        logger.info("Initial rank: %d; total additional ranks: %d", self.peft_config.init_r, total_additional_rank)

        num_rounds = math.ceil(total_additional_rank / rank_per_round)

        total_incre_step = self.peft_config.growth_interval * num_rounds

        logger.info("Growing %d ranks per round  => %d rounds of growth (each %d steps long)", rank_per_round, num_rounds, self.peft_config.growth_interval)

        logger.info(
            "Total growth phase steps: %d; fraction of total steps: %.2f",
            total_incre_step,
            total_incre_step / total_steps,
        )

        new_params = model.setup_reserve_ranks()

        self.add_new_param(*new_params, reserve=True)
        self.update_param_groups(optimizer)

        self.peft_config.rank_pattern = model.get_rank_pattern(self.adapter_name)

    def reinitialize(self, model, optimizer):
        self.new_reserve_params.clear()
        self.new_main_params.clear()
        self.old_reserve_params.clear()

        # Clear the optimizer groups
        optimizer.state.clear()
        optimizer.param_groups.clear()

        # Reinitialize weights
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                layer.reset_lora_parameters(
                    adapter_name=self.adapter_name,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )

                # Add reinitialized parameters to the groups
                for i, non_reserve in enumerate(layer.rank_pattern[self.adapter_name]):
                    if non_reserve:
                        params = (
                            layer.lora_A[self.adapter_name][i],
                            layer.lora_B[self.adapter_name][i],
                            layer.lora_E[self.adapter_name][i],
                        )
                    elif self.peft_config.advance_learn:
                        params = (
                            layer.lora_A[self.adapter_name][i],
                            layer.lora_B[self.adapter_name][i],
                        )
                    else:
                        continue

                    self.add_new_param(*params, reserve=not non_reserve)

    def add_new_param(self, *params, reserve: bool = False):
        if reserve:
            self.new_reserve_params.extend(params)
        else:
            self.new_main_params.extend(params)

    def make_param_main(self, param, *, already_trained: bool = False):
        if not already_trained:
            param.requires_grad = True
            self.add_new_param(param, reserve=False)

        elif self.reserve_separate_groups:
            self.old_reserve_params.append(param)
            self.add_new_param(param, reserve=False)

        else:
            # Nothing to be done here, we can continue to train the paramter as before
            pass

    @staticmethod
    def delete_param_from_groups(optimizer, param):
        for group_index, group in enumerate(optimizer.param_groups):
            param_index = None

            for i, p in enumerate(group["params"]):
                if p is param:
                    param_index = i
                    break

            if param_index is None:
                continue

            try:
                del optimizer.state[param]
            except KeyError:
                logger.warning("%d state keys.", len(optimizer.state))
                logger.warning(
                    "Found parameter (%d, %s) in group %d, but not in optimizer state.",
                    param_index,
                    str(param.shape),
                    group_index
                )
                raise

            del group["params"][param_index]

            if len(group["params"]) == 0:
                del optimizer.param_groups[group_index]

            return

        msg = f"Could not find parameter ({param.shape}) in `optimizer.param_groups`."
        raise ValueError(msg)

    def update_param_groups(self, optimizer, **kw):
        logger.info("Removing %d reserve rank parameters", len(self.old_reserve_params))
        logger.info("Adding %d reserve rank parameters", len(self.new_reserve_params))
        logger.info("Adding %d main rank parameters", len(self.new_main_params))

        for param in self.old_reserve_params:
            param_index = None

            for i, p in enumerate(self.new_reserve_params):
                if p is param:
                    param_index = i
                    break

            if param_index is not None:
                # Not yet added to the optimizer
                # Note that even if we chose to add new params right away,
                # the state may not have been initialized yet.
                del self.new_reserve_params[param_index]
            else:
                self.delete_param_from_groups(optimizer, param)

        if self.reserve_separate_groups:
            # Create new group sepcifically for the main params
            if len(self.new_main_params) > 0:
                optimizer.add_param_group(
                    {
                        "params": self.new_main_params,
                        "weight_decay": self.weight_decay,
                        **kw
                    }
                )

            # Create group for the reserve ranks
            if len(self.new_reserve_params) > 0:
               group = {
                   "params": self.new_reserve_params,
                   "weight_decay": self.reserve_weight_decay,
                   "constant": self.reserve_constant_lr,
                   **kw
               }

               if self.reserve_lr is not None:
                   group["initial_lr"] = self.reserve_lr

               if self.reserve_betas is not None:
                   group["betas"] = tuple(self.reserve_betas)

               optimizer.add_param_group(group)

        else:
            # We can throw both of the lists into one group
           params = self.new_reserve_params + self.new_main_params

           if len(params) > 0:
                optimizer.add_param_group({
                        "params": params,
                        "weight_decay": self.weight_decay,
                        **kw
                    })

        self.old_reserve_params.clear()
        self.new_reserve_params.clear()
        self.new_main_params.clear()

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
        """Return the scores for reserve ranks of each layer."""
        module_scores: dict[str, torch.Tensor] = {}

        # Calculate the importance score for each sub matrix
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                if self.random_selection:
                    module_scores[n] = torch.rand(
                        self.peft_config.reserve_ranks,
                        device="cpu"
                    )

                elif self.target_rank_pattern is not None:
                    target_rank_key = get_pattern_key(self.target_rank_pattern, n)
                    target_rank = self.target_rank_pattern.get(target_rank_key, 0)

                    current_rank = layer.r[self.adapter_name]
                    reserve_ranks = self.reserve_ranks
                    remaining_ranks = target_rank - current_rank

                    module_scores[n] = torch.full((self.peft_config.reserve_ranks, ), -1, device="cpu")

                    if remaining_ranks > 0:
                        mod = remaining_ranks % reserve_ranks
                        div = remaining_ranks // reserve_ranks

                        module_scores[n][:mod] = div + 1

                        if div > 0:
                            module_scores[n][mod:] = div

                        module_scores[n] += 0.5 * torch.rand(
                            self.peft_config.reserve_ranks,
                            device="cpu",
                        )

                elif self.ignore_uncertainty:
                    reserve = ~torch.tensor(layer.rank_pattern[self.adapter_name])
                    module_scores[n] = self.exp_avg_grad[n][reserve].abs().cpu()
                else:
                    reserve = ~torch.tensor(layer.rank_pattern[self.adapter_name])
                    module_scores[n] = (self.exp_avg_grad[n][reserve].abs() * self.exp_avg_unc[n][reserve]).cpu()

        return module_scores

    def increase_layer_rank(self, layer: SVDLinear, ranks_to_add: Iterable[int]) -> None:
        """Add the selected ranks to the layer.

        Args:
            ranks_to_add: specifies either the number of ranks to add
            (i.e. all reserve ranks from the back of the parameter list)
            or a boolean mask specifying which ranks to add.

        Returns:
            The parameters that need to be added to the optimizer.
        """
        lora_E = layer.lora_E[self.adapter_name]

        num_added: int = len(ranks_to_add)

        layer.r[self.adapter_name] += num_added
        self.total_current_rank += num_added

        assert len(layer.rank_pattern[self.adapter_name]) == len(lora_E)

        reserve_rank_indices = tuple(i for i, non_r in enumerate(layer.rank_pattern[self.adapter_name]) if not non_r)

        for reserve_index in ranks_to_add:
            i = reserve_rank_indices[reserve_index]

            assert not layer.rank_pattern[self.adapter_name][i]
            assert not layer.lora_E[self.adapter_name][i].requires_grad

            self.make_param_main(layer.lora_E[self.adapter_name][i], already_trained=False)
            self.make_param_main(layer.lora_A[self.adapter_name][i], already_trained=self.peft_config.advance_learn)
            self.make_param_main(layer.lora_B[self.adapter_name][i], already_trained=self.peft_config.advance_learn)

            layer.rank_pattern[self.adapter_name][i] = True

        self.add_new_param(*layer.add_reserve_ranks(self.adapter_name, num_added), reserve=True)

    @staticmethod
    def get_topk_ranks(module_scores: dict[str, torch.Tensor], k: int) -> tuple[defaultdict[str, tuple[int, ...]], float]:
        ranks_to_add: defaultdict[str, list[int, ...]] = defaultdict(list)

        offsets: list[tuple[str, int]] = []
        current_offset: int = 0
        all_scores = []

        for n, scores in module_scores.items():
            offsets.append((n, current_offset))
            current_offset += scores.size(0)

            all_scores.append(scores)

        values, indices  = torch.topk(torch.cat(all_scores), k, sorted=False)
        increase_threshold = values.min().item() # values[-1].item()

        offset_iterator = reversed(offsets)

        layer, offset = next(offset_iterator)

        logging.info(str(indices))
        logging.info(str(offsets))

        for idx in sorted(indices.tolist(), reverse=True):
            while offset > idx:
                layer, offset = next(offset_iterator)

            ranks_to_add[layer].append(idx - offset)

        return ranks_to_add, increase_threshold

    def increase_to_target_rank(self, model, optimizer):
        # Calculate the increasing threshold
        k = min(self.top_h * self.reserve_ranks, self.total_target_rank - self.total_current_rank)

        if not k > 0:
            return float("Inf")

        module_scores = self.retrieve_scores(model)
        ranks_to_add, increase_threshold = self.get_topk_ranks(module_scores, k)

        logger.info(
            "Increase threshold: %e", increase_threshold
        )

        metrics = {}
        with torch.no_grad():
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    if len(ranks_to_add[n]) > 0:

                        self.increase_layer_rank(layer, sorted(ranks_to_add[n]))

                        assert len(ranks_to_add[n]) <= self.reserve_ranks

                        self.peft_config.rank_pattern[n] = layer.rank_pattern
                        logger.info("The lora parameters rank of %s increased by %d", n, len(ranks_to_add[n]))

                    # log metrics
                    metrics[f"num_rank/{n}"] = layer.r[self.adapter_name]

            if self.total_current_rank >= self.total_target_rank:
                self.old_reserve_params.extend(model.drop_reserve(self.adapter_name))
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


            if (global_step == -1) or (global_step >= growth_delay and (global_step - growth_delay) % self.peft_config.growth_interval == 0):
                logger.info(
                    "Growth step at step %d (growth delay: %d (%s), growth interval: %d)",
                    global_step,
                    growth_delay,
                    "set in config" if self.peft_config.growth_delay is not None else "warmup",
                    self.peft_config.growth_interval,
                )

                remaining_steps = self.total_steps - global_step

                self.increase_to_target_rank(
                    model, optimizer,
                )

                param_group_kws={
                    "initial_step": global_step,
                    "warmup_steps": training_args.get_warmup_steps(remaining_steps),
                    "remaining_steps": remaining_steps,
                }

                if self.reinit_after_growth:
                    self.reinitialize(
                        model=model,
                        optimizer=optimizer
                    )

                self.update_param_groups(
                    optimizer,
                    **param_group_kws
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
