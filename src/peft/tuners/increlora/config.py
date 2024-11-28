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
from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class IncreLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a
    [`~peft.IncreLora`].

    It should be noted that in any case top_h * reserve_ranks total
    ranks are addeed per round. In the original scoring scheme,
    reserve_ranks ranks are added in top_h modules. In the alternative
    scoring scheme, the total ranks are distributed across at minimum
    top_h modules but potentially as many as top_h * reserve_ranks.
    I.e. as many as reserve_ranks ranks are added.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Initial Lora matrix dimension."})

    deltaT: int = field(default=1000, metadata={"help": "The time internval between two budget allocations."})

    top_h: int = field(default=5, metadata={"help": "The number of modules selected."})
    reserve_ranks: int = field(
        default=1,
        metadata={
            "help": (
                "The number of ranks to add per selected module. If `alternative_scoring` is "
                "enabled, it determines the number of reserve ranks that are added to each module."
            )
        },
    )

    alternative_scoring: bool = field(
        default=False, metadata={"help": "Whether to use the alternative scoring scheme."}
    )
    orthonormalize: bool = field(default=False, metadata={"help": "Whether to enforce orthonormalization."})

    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict[str, list[bool]]] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        self.peft_type = PeftType.INCRELORA

        if self.use_dora:
            raise ValueError(f"{self.peft_type} does not support DoRA.")

        if self.loftq_config:
            raise ValueError(f"{self.peft_type} does not support LOFTQ.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        # Check if 'r' has been set to a non-default value
        if self.r != 8:  # 8 is the default value for 'r' in LoraConfig
            warnings.warn(
                "Note that `r` is not used in AdaLora and will be ignored."
                "If you intended to set the initial rank, use `init_r` instead."
            )
