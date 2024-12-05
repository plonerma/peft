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

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)
from peft.utils.integrations import gather_params_ctx

from .layer import IncreLoraLayer, SVDLinear


class IncreLoraModel(LoraModel):
    """
    Creates IncreLora model from a pretrained transformers model. Paper:
    https://arxiv.org/abs/2308.12043

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IncreLoraConfig`]): The configuration of the AdaLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The IncreLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM >>> from peft import LoraConfig, AdaLoraModel, AdaLoraConfig
        >>> config = AdaLoraConfig(
                peft_type="INCRELORA", task_type="SEQ_2_SEQ_LM", init_r=12, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = AdaLoraModel(model, config, "default")

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    # Note: don't redefine prefix here, it should be inherited from LoraModel
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "init_r": lora_config.init_r,
            "reserve_ranks": lora_config.reserve_ranks,
            "alternative_scoring": lora_config.alternative_scoring,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise NotImplementedError

        # If it is not an IncreLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, IncreLoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise NotImplementedError

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = SVDLinear(target, adapter_name, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def add_weighted_adapter(self, *args, **kwargs):
        """This method is not supported for IncreLoRA, use LoRA instead."""
        raise TypeError(f"{self.__class__.__name__} does not support add_weighted_adapter method.")

    def resize_modules_by_rank_pattern(self, *, adapter_name, rank_pattern):
        lora_config = self.peft_config[adapter_name]

        if rank_pattern is None:
            rank_pattern = lora_config.rank_pattern

        for name, pattern in rank_pattern.items():
            if not isinstance(pattern, list):
                raise ValueError("Unexpected type of is_reserve_rank")

            parts = name.split(".")

            if adapter_name in parts:
                parts = parts[:-1]

            if parts[0] == "model":
                parts = parts[1:]

            key = ".".join(parts)
            _, target, _ = _get_submodules(self.model, key)

            lora_E_weights = target.lora_E[adapter_name]
            lora_A_weights = target.lora_A[adapter_name]
            lora_B_weights = target.lora_B[adapter_name]

            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

            add_r = len(pattern)

            if lora_config.init_r > 0:
                add_r -= 1
                print("rp", pattern)
                print("!", lora_config.init_r, sum(pattern))
                ranknum = lora_config.init_r + sum(pattern) - 1
            else:
                ranknum = sum(pattern)
            print("=", ranknum)

            target.add_reserve_ranks(adapter_name, add_r)

            with torch.no_grad():
                target.lora_E[adapter_name][0].copy_(lora_E_weights[0])
                target.lora_A[adapter_name][0].copy_(lora_A_weights[0])
                target.lora_B[adapter_name][0].copy_(lora_B_weights[0])

            target.r[adapter_name] = ranknum
            target.rank_pattern[adapter_name] = pattern

    def get_rank_pattern(self, adapter_name: str):
        rank_pattern: dict[str, list[bool]] = {}
        for n, layer in self.named_modules():
            if isinstance(layer, SVDLinear):
                rank_pattern[n] = layer.rank_pattern[adapter_name]

        return rank_pattern
