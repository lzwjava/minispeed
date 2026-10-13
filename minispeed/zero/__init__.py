from typing import List

from ..flat import FlatParamGroup
from .stage_1_and_2 import ZeROOptimizer
from .stage_3 import ZeROStage3Optimizer


def build_zero_optimizer(
    flat_groups: List[FlatParamGroup],
    option_groups: List[dict],
    zero_stage: int = 0,
) -> ZeROOptimizer:
    """Optimizer factory: stage 3 partitions the parameters as well."""
    if zero_stage >= 3:
        return ZeROStage3Optimizer(flat_groups, option_groups)
    return ZeROOptimizer(flat_groups, option_groups, zero_stage=zero_stage)


__all__ = ["ZeROOptimizer", "ZeROStage3Optimizer", "build_zero_optimizer"]