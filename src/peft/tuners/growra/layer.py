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

import math
import warnings
from functools import partial
from itertools import chain
from typing import Any, List, Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


class GrowRALayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = (
        "lora_A",
        "lora_B",
        "lora_E",
        "lora_embedding_A",
        "lora_embedding_B",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "rank_pattern", "target_r")

    rank_pattern: dict[str, list[bool]]

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

        self.rank_pattern = {}
        self.target_r = {}

    def update_layer(self, *, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, target_r):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.target_r[adapter_name] = target_r

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.ParameterList([])
        self.lora_E[adapter_name] = nn.ParameterList([])
        self.lora_B[adapter_name] = nn.ParameterList([])

        self.rank_pattern[adapter_name] = []

        if r > 0:
            self.lora_A[adapter_name].append(nn.Parameter(torch.empty(r, self.in_features)))
            self.lora_E[adapter_name].append(nn.Parameter(torch.empty(r)))
            self.lora_B[adapter_name].append(nn.Parameter(torch.empty(self.out_features, r)))
            self.rank_pattern[adapter_name].append(True)

        # The current rank
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        self.init_lora_weights = init_lora_weights
        if init_lora_weights.lower() == "increlora":
            if adapter_name in self.lora_A.keys():
                for p in self.lora_E[adapter_name]:
                    nn.init.zeros_(p)

                for p in chain(self.lora_A[adapter_name], self.lora_B[adapter_name]):
                    nn.init.normal_(p, mean=0.0, std=0.02)
        elif init_lora_weights.lower() == "lora":
            if adapter_name in self.lora_A.keys():
                for p in self.lora_E[adapter_name]:
                    nn.init.ones_(p)

                for p in self.lora_A[adapter_name]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

                for p in self.lora_B[adapter_name]:
                    nn.init.zeros_(p)

        elif init_lora_weights.lower() == "growra":
            if adapter_name in self.lora_A.keys():
                for p in self.lora_E[adapter_name]:
                    nn.init.zeros_(p)

                for p in self.lora_A[adapter_name]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5), nonlinearity="linear", mode="fan_in")

                for p in self.lora_B[adapter_name]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5), nonlinearity="linear", mode="fan_out")

        else:
            msg = f"Weight init method `{init_lora_weights}` unkown."
            raise ValueError(msg)

    def _move_adapter_to_device_of_base_layer(self, adapter_name):
        device = self.base_layer.weight.device
        self.lora_A[adapter_name] = self.lora_A[adapter_name].to(device)
        self.lora_E[adapter_name] = self.lora_E[adapter_name].to(device)
        self.lora_B[adapter_name] = self.lora_B[adapter_name].to(device)


class GrowRAComputation(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, input, a, b, e, reserve, ignore_reserve: bool, scale_grads: bool):
        ctx.scale_all = scale_grads
        ctx.ignore_reserve = ignore_reserve

        if ignore_reserve:
            a = a[~reserve]
            e = e[~reserve]
            b = b[:, ~reserve]

        # Here, it would be faster to compute (e*a) first, but we need i_a for the backward pass as well
        i_a = torch.matmul(input, a.mT)
        pre_b = e * i_a

        z = torch.matmul(pre_b, b.mT)

        ctx.save_for_backward(input, a, b, e, i_a, pre_b, reserve)

        return z

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        with torch.no_grad():
            input, a, b, e, i_a, pre_b, reserve = ctx.saved_tensors


            grad_pre_b = torch.matmul(grad_output, b)

            grad_input = torch.matmul(grad_pre_b[..., ~reserve], (torch.diag(e[~reserve]) @ a[~reserve]))

            grad_a, grad_b, grad_e = None, None, None

            # Reduce batch/sequence dimensions for efficient computation
            input = input.view(-1, input.size(-1))
            grad_output = grad_output.view(-1, grad_output.size(-1))
            i_a = i_a.view(-1, i_a.size(-1))

            grad_pre_b = grad_pre_b.view(-1, pre_b.size(-1))

            if ctx.scale_all:
                e = torch.sign(e.detach())

            else:
                e = torch.empty_like(e).copy_(e)

            e[reserve] = 1.0

            if ctx.needs_input_grad[1]:
                grad_a = torch.mm((grad_pre_b * e).mT, input)

            if ctx.needs_input_grad[2]:
                grad_b = torch.mm(grad_output.mT, (i_a * e))

            if ctx.needs_input_grad[3]:
                grad_e = torch.mm(grad_pre_b.mT, i_a)

            return grad_input, grad_a, grad_b, grad_e, None, None, None



class SVDLinear(nn.Module, GrowRALayer):
    EPS = 1e-5

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        init_r: int = 0,
        target_r: int = 0,
        advance_learn: bool = True,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        dynamic_scaling: bool = True,
        scale_all_grads: bool = False,
        reserve_rank_scoring: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        GrowRALayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.advance_learn = advance_learn
        self.dynamic_scaling = dynamic_scaling
        self.scale_all_grads = scale_all_grads
        self.reserve_rank_scoring = reserve_rank_scoring

        self.hook_handle = None

        self.e_grad = None

        self.update_layer(
            adapter_name=adapter_name,
            r=init_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            target_r=target_r
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_scaling_coeff(self, adapter) -> float:
        if self.dynamic_scaling:
            return self.scaling[adapter] / max(self.r[adapter], 1)
        else:
            return self.scaling[adapter] / self.target_r[adapter]

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if len(self.lora_A[adapter]) == 0:
            return torch.zeros_like(self.base_layer.weight)

        lora_A = torch.cat(tuple(self.lora_A[adapter]), 0)
        lora_B = torch.cat(tuple(self.lora_B[adapter]), 1)
        lora_E = torch.cat(tuple(self.lora_E[adapter]), 0)
        return transpose(lora_B @ (torch.diag(lora_E) @ lora_A), self.fan_in_fan_out) * self.get_scaling_coeff(adapter)

    def backward_hook(self, param, grad, index=None):
        """Note that this is the pre-accumulation gradient hook.

        If gradients are accumulated, we also need to do this here.
        """
        if self.reserve_rank_scoring:
            if self.e_grad is None:
                self.e_grad = grad
            else:
                self.e_grad += grad

        else:
            self.ipt = (param.detach() * grad.detach()).abs().mean()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys() or len(self.lora_A[active_adapter]) == 0:
                    continue

                dropout = self.lora_dropout[active_adapter]
                x = x.to(self.lora_A[active_adapter][0].dtype)

                if self.training:
                    x = dropout(x)

                    if self.reserve_rank_scoring:
                        lora_A = torch.cat(tuple(self.lora_A[active_adapter]), 0)
                        lora_B = torch.cat(tuple(self.lora_B[active_adapter]), 1)
                        lora_E = torch.cat(tuple(self.lora_E[active_adapter]), 0)
                        reserve = self.get_reserve_mask(active_adapter)

                        lora_E.requires_grad_(True)
                        lora_E.register_hook(partial(self.backward_hook, lora_E))

                        r = GrowRAComputation.apply(
                            x, lora_A, lora_B, lora_E, reserve, False, self.scale_all_grads,
                        ) * self.get_scaling_coeff(active_adapter)

                        result += r

                    else:
                        w = self.get_delta_weight(active_adapter)
                        if torch.is_grad_enabled():
                            if self.hook_handle is not None:
                                self.hook_handle.remove()
                            w.requires_grad_(True)
                            w.register_hook(partial(self.backward_hook, w))
                        result += x @ w.T
                else:
                    lora_A = torch.cat(tuple(self.lora_A[active_adapter]), 0)
                    lora_B = torch.cat(tuple(self.lora_B[active_adapter]), 1)
                    lora_E = torch.cat(tuple(self.lora_E[active_adapter]), 0)
                    reserve = self.get_reserve_mask(active_adapter)

                    r = GrowRAComputation.apply(
                        x, lora_A, lora_B, lora_E, reserve, True, self.scale_all_grads,
                    ) * self.get_scaling_coeff(active_adapter)

                    result += r

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "growra." + rep

    def add_reserve_ranks(self, adapter_name: str, add_r: int) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for _ in range(add_r):
            e = nn.Parameter(
                torch.full((1, ), self.EPS),
                requires_grad=False,
            )
            a = nn.Parameter(torch.empty((1, self.in_features)), requires_grad=self.advance_learn)
            b = nn.Parameter(torch.empty((self.out_features, 1)), requires_grad=self.advance_learn)

            if self.init_lora_weights.lower() == "increlora":
                e.data.fill_(1e-5)
                nn.init.normal_(a, mean=0.0, std=0.02)
                nn.init.normal_(b, mean=0.0, std=0.02)

            elif self.init_lora_weights.lower() == "lora":

                nn.init.ones_(e)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                nn.init.zeros_(b)

            elif self.init_lora_weights.lower() == "growra":
                nn.init.zeros_(e)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5), nonlinearity="linear", mode="fan_in")
                nn.init.kaiming_uniform_(b, a=math.sqrt(5), nonlinearity="linear", mode="fan_out")

            else:
                msg = f"Weight init method `{self.init_lora_weights}` unkown."
                raise ValueError(msg)

            self.lora_E[adapter_name].append(e)
            self.lora_A[adapter_name].append(a)
            self.lora_B[adapter_name].append(b)

            self.rank_pattern[adapter_name].append(False)

            if self.advance_learn:
                parameters.extend((a, b))

        self._move_adapter_to_device_of_base_layer(adapter_name)

        return parameters

    def get_reserve_mask(self, adapter_name):
        return torch.cat(
            [
                torch.full((e.size(0),), not r)
                for r, e in zip(self.rank_pattern[adapter_name], self.lora_E[adapter_name])
            ]
        )

    def get_rank(self, adapter_name, *, include_reserve=False) -> int:
        return sum(
            (
                e.size(0)
                for r, e in zip(self.rank_pattern[adapter_name], self.lora_E[adapter_name])
                if r or include_reserve
            )
        )

    def drop_reserve(self, adapter_name) -> list[nn.Parameter]:
        dropped_params = []

        if self.advance_learn:
            # If a and b were learned before, they can now be removed from the optimizer
            dropped_params.extend((
                a for r, a in zip(self.rank_pattern[adapter_name], self.lora_A[adapter_name]) if not r
            ))
            dropped_params.extend((
                b for r, b in zip(self.rank_pattern[adapter_name], self.lora_B[adapter_name]) if not r
            ))

        self.lora_A[adapter_name] = nn.ParameterList(
            [a for r, a in zip(self.rank_pattern[adapter_name], self.lora_A[adapter_name]) if r]
        )
        self.lora_B[adapter_name] = nn.ParameterList(
            [b for r, b in zip(self.rank_pattern[adapter_name], self.lora_B[adapter_name]) if r]
        )
        self.lora_E[adapter_name] = nn.ParameterList(
            [e for r, e in zip(self.rank_pattern[adapter_name], self.lora_E[adapter_name]) if r]
        )

        # Delete reserve ranks from rank pattern (modfiy list in place to keep other references intact)
        for i in reversed(range(len(self.rank_pattern[adapter_name]))):
            if not self.rank_pattern[adapter_name][i]:
                del self.rank_pattern[adapter_name][i]

        return dropped_params
