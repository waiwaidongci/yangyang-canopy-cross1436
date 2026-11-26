"""Compatibility import path for roll grouping rules."""

from .grouping import *  # noqa: F403
from .grouping import (
    GroupingError,
    GroupingPlan,
    Partition,
    RollGroup,
    build_roll_groups,
    choose_auto_partition,
    coerce_roll_map,
    fingerprint_grouping,
    groups_from_lot_lists,
    plan_grouping,
    select_holdout_partition,
    snapshot_bridges_sides,
)

__all__ = [
    "GroupingError", "GroupingPlan", "Partition", "RollGroup",
    "build_roll_groups", "choose_auto_partition", "coerce_roll_map",
    "fingerprint_grouping", "groups_from_lot_lists", "plan_grouping",
    "select_holdout_partition", "snapshot_bridges_sides",
]
