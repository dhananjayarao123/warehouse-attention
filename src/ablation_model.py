"""Compatibility wrapper for the controlled scaled/unscaled ablation.

The ablation deliberately reuses the same WarehouseAttentionModel as the main
warehouse experiment. The only controlled change is the attention scaling flag.
"""

from .warehouse_model import WarehouseAttentionModel

__all__ = ["WarehouseAttentionModel"]
