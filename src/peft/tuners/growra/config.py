# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class GrowRAConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a
    [`~peft.GrowRA`].

    It should be noted that in any case top_h * reserve_ranks total
    ranks are addeed per round. In the original scoring scheme,
    reserve_ranks ranks are added in top_h modules. In the alternative
    scoring scheme, the total ranks are distributed across at minimum
    top_h modules but potentially as many as top_h * reserve_ranks.
    I.e. as many as reserve_ranks ranks are added.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Initial Lora matrix dimension."})

    growth_interval: int = field(default=1000, metadata={"help": "The time internval between two budget allocations."})

    num_top_modules: int = field(default=5, metadata={"help": "The number of modules selected."})

    reserve_ranks: int = field(
        default=1,
        metadata={
            "help": (
                "The number of ranks to add per selected module. If `alternative_scoring` is "
                "enabled, it determines the number of reserve ranks that are added to each module."
            )
        },
    )

    orthonormalize: bool = field(default=False, metadata={"help": "Whether to enforce orthonormalization."})
    orthonormalize_reserve_only: bool = field(
        default=False, metadata={"help": "Whether to enforce orthonormalization only on the reserve ranks."}
    )
    orthonormalize_ignore_non_reserve: bool = field(
        default=False, metadata={"help": "Whether to enforce orthonormalization only on the reserve ranks."}
    )

    normalize: bool = field(default=False, metadata={"help": "Whether to enforce normalization."})
    normalize_reserve_only: bool = field(
        default=False, metadata={"help": "Whether to enforce normalization only on the reserve ranks."}
    )

    dynamic_scaling: bool = field(
        default=True,
        metadata={"help": "Whether to scale adapter contribution based on current rank (instead of target_r)."},
    )

    growra_scale_all_grads: bool = field(
        default=True,
        metadata={"help": "Whether to scale A B gradients by the inverse of the gradient."},
    )

    init_lora_weights: Literal["lora", "increlora"] = "lora"

    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict[str, list[bool]]] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        self.peft_type = PeftType.GROWRA

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
