"""Evaluate the two predictions on exactly the same unseen groups."""

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from .features import CrossModel, Score, pair_of
from .records import SeamRecord


@dataclass(frozen=True)
class ScoredSeam:
    row: SeamRecord
    score: Score


@dataclass(frozen=True)
class BatchErrors:
    lot_id: str
    rows: int
    crossed: int
    main_mae: float
    cross_mae: float

    @property
    def gain(self) -> float:
        return self.main_mae - self.cross_mae


@dataclass(frozen=True)
class GroupErrors:
    group_id: str
    lot_ids: tuple[str, ...]
    rows: int
    crossed: int
    main_mae: float
    cross_mae: float

    @property
    def gain(self) -> float:
        return self.main_mae - self.cross_mae

    # Compatibility for callers that treat a group as a macro batch row.
    @property
    def lot_id(self) -> str:
        return self.group_id


@dataclass(frozen=True)
class HoldoutEvaluation:
    records: list[ScoredSeam]
    batches: list[BatchErrors]
    groups: list[GroupErrors] | None = field(default=None, compare=False)

    @property
    def macro_units(self) -> list[BatchErrors | GroupErrors]:
        return self.groups if self.groups is not None else self.batches

    @property
    def main_mae(self) -> float:
        return mean(unit.main_mae for unit in self.macro_units)

    @property
    def cross_mae(self) -> float:
        return mean(unit.cross_mae for unit in self.macro_units)

    @property
    def gain(self) -> float:
        return self.main_mae - self.cross_mae

    @property
    def coverage(self) -> float:
        return 100 * sum(batch.crossed for batch in self.batches) / len(self.records)

    @property
    def group_main_mae(self) -> float:
        return self.main_mae

    @property
    def group_cross_mae(self) -> float:
        return self.cross_mae

    @property
    def group_gain(self) -> float:
        return self.gain


def score_records(model: CrossModel, rows: list[SeamRecord]) -> list[ScoredSeam]:
    return [ScoredSeam(row, model.predict(pair_of(row))) for row in rows]


def _errors(items: list[ScoredSeam]) -> tuple[int, int, float, float]:
    main_errors: list[float] = []
    cross_errors: list[float] = []
    crossed = 0
    for item in items:
        actual = item.row["wrinkle_mm"]
        assert actual is not None
        main_errors.append(abs(actual - item.score.main_mm))
        cross_errors.append(abs(actual - item.score.cross_mm))
        crossed += int(item.score.uses_cross)
    return len(items), crossed, mean(main_errors), mean(cross_errors)


def evaluate_holdout(
    model: CrossModel,
    holdout: list[SeamRecord],
    groups: dict[str, str] | list | tuple | None = None,
    *,
    group_lots: dict[str, list[str] | tuple[str, ...]] | None = None,
    lot_group: dict[str, str] | None = None,
    roll_groups: list | tuple | None = None,
) -> HoldoutEvaluation:
    """Score holdout records and macro-average by cloth-roll group.

    Without an explicit grouping each lot is its own group, preserving the
    original lot-equal weighted calculation exactly.
    """

    scored = score_records(model, holdout)
    grouped: dict[str, list[ScoredSeam]] = defaultdict(list)
    for item in scored:
        grouped[item.row["lot_id"]].append(item)

    batches: list[BatchErrors] = []
    for lot, items in sorted(grouped.items()):
        rows, crossed, main_mae, cross_mae = _errors(items)
        batches.append(BatchErrors(lot, rows, crossed, main_mae, cross_mae))

    if lot_group is None and isinstance(groups, dict):
        # Accept either lot -> group id or group id -> lot list.
        if groups and all(isinstance(value, (list, tuple, set)) for value in groups.values()):
            group_lots = {key: tuple(value) for key, value in groups.items()}
        else:
            lot_group = dict(groups)
    elif group_lots is None and roll_groups is None and groups is not None:
        # Positional list/tuple of group ids/lot lists/domain RollGroup objects.
        roll_groups = groups
    if group_lots is None and roll_groups is not None:
        group_lots = {}
        for index, group in enumerate(roll_groups, start=1):
            if isinstance(group, dict):
                group_id = group.get("group_id") or group.get("id") or f"RG{index:02d}"
                lots = group.get("lot_ids") or group.get("lots") or []
            else:
                group_id = getattr(group, "group_id", f"RG{index:02d}") or f"RG{index:02d}"
                lots = getattr(group, "lot_ids", group)
            group_lots[group_id] = tuple(lots)
    if group_lots is not None and lot_group is None:
        lot_group = {lot: group_id for group_id, lots in group_lots.items() for lot in lots}

    if lot_group is None:
        # Convenience for direct domain calls whose rows carry a roll/group id.
        roll_keys = (
            "roll_id", "roll_number", "roll_no", "cloth_roll_id",
            "roll_group_id", "group_id", "roll_group",
        )
        inferred: dict[str, str] = {}
        for item in scored:
            for key in roll_keys:
                value = item.row.get(key) if isinstance(item.row, dict) else None
                if value not in (None, ""):
                    inferred[item.row["lot_id"]] = str(value)
                    break
        if inferred:
            lot_group = inferred

    if lot_group is None:
        evaluation_groups = [
            GroupErrors(batch.lot_id, (batch.lot_id,), batch.rows, batch.crossed,
                        batch.main_mae, batch.cross_mae)
            for batch in batches
        ]
    else:
        # Ensure every holdout lot has a group even if caller supplied a partial map.
        lot_group = {lot: lot_group.get(lot, lot) for lot in grouped}
        by_group: dict[str, list[ScoredSeam]] = defaultdict(list)
        group_lot_ids: dict[str, set[str]] = defaultdict(set)
        for item in scored:
            group_id = lot_group[item.row["lot_id"]]
            by_group[group_id].append(item)
            group_lot_ids[group_id].add(item.row["lot_id"])
        evaluation_groups = []
        for group_id in sorted(by_group):
            items = by_group[group_id]
            rows, crossed, main_mae, cross_mae = _errors(items)
            evaluation_groups.append(GroupErrors(
                group_id, tuple(sorted(group_lot_ids[group_id])), rows, crossed, main_mae, cross_mae,
            ))

    return HoldoutEvaluation(scored, batches, evaluation_groups)


def macro_mae(evaluation: HoldoutEvaluation) -> tuple[float, float, float]:
    """Return group-equal main MAE, cross MAE and gain."""

    return evaluation.main_mae, evaluation.cross_mae, evaluation.gain


# Descriptive aliases for direct acceptance calls.
evaluate_by_roll_groups = evaluate_holdout
evaluate_by_groups = evaluate_holdout
group_macro_mae = macro_mae
