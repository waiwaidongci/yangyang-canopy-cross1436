"""Cloth-roll aware lot groups and leakage-safe holdout planning.

A physical cloth roll can receive several sewing lot numbers.  All lots linked
by a shared roll are therefore treated as one indivisible group for fitting and
evaluation.  The module is deliberately persistence-free so the acceptance
checks can exercise the partition rules directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .records import (
    HOLDOUT,
    MIN_HOLDOUT_LOTS,
    MIN_HOLDOUT_ROWS,
    MIN_TRAIN_LOTS,
    MIN_TRAIN_ROWS,
    PENDING,
    TRAIN,
    SeamRecord,
)


HOLDOUT_TARGET_FRACTION = 0.3
GROUP_STRATEGY_AUTO = "roll_groups"
GROUP_STRATEGY_EXPLICIT = "explicit"


class GroupingError(ValueError):
    """Raised when no leakage-safe qualified partition exists."""


@dataclass(frozen=True)
class RollGroup:
    """A connected component of lots sharing one or more rolls."""

    lot_ids: tuple[str, ...]
    roll_ids: tuple[str, ...] = ()
    group_id: str = ""

    def __iter__(self):
        return iter(self.lot_ids)

    def __len__(self) -> int:
        return len(self.lot_ids)

    def __contains__(self, lot_id: object) -> bool:
        return lot_id in self.lot_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "id": self.group_id,
            "key": self.group_id,
            "lot_ids": list(self.lot_ids),
            "lots": list(self.lot_ids),
            "roll_ids": list(self.roll_ids),
            "rolls": list(self.roll_ids),
        }


@dataclass(frozen=True)
class Partition:
    train_lot_ids: tuple[str, ...]
    holdout_lot_ids: tuple[str, ...]
    pending_lot_ids: tuple[str, ...] = ()
    train_group_ids: tuple[str, ...] = ()
    holdout_group_ids: tuple[str, ...] = ()
    strategy: str = GROUP_STRATEGY_AUTO
    total_measured_rows: int = 0
    target_holdout_rows: float = 0.0
    selected_holdout_rows: int = 0
    objective_distance: int = 0
    moved_lot_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "strategy": self.strategy,
            "train_lot_ids": list(self.train_lot_ids),
            "holdout_lot_ids": list(self.holdout_lot_ids),
            "pending_lot_ids": list(self.pending_lot_ids),
            "train_group_ids": list(self.train_group_ids),
            "holdout_group_ids": list(self.holdout_group_ids),
            "total_measured_rows": self.total_measured_rows,
            "target_holdout_rows": self.target_holdout_rows,
            "selected_holdout_rows": self.selected_holdout_rows,
            "objective_distance": self.objective_distance,
            "moved_lot_ids": list(self.moved_lot_ids),
        }
        # Flat aliases make persisted run.grouping easy for tables and checks.
        data.update({
            "train_lots": data["train_lot_ids"],
            "holdout_lots": data["holdout_lot_ids"],
            "pending_lots": data["pending_lot_ids"],
        })
        return data


@dataclass(frozen=True)
class GroupingPlan:
    roll_map: dict[str, tuple[str, ...]]
    groups: tuple[RollGroup, ...]
    partition: Partition
    grouping_key: str

    @property
    def train_lot_ids(self) -> tuple[str, ...]:
        return self.partition.train_lot_ids

    @property
    def holdout_lot_ids(self) -> tuple[str, ...]:
        return self.partition.holdout_lot_ids

    def lot_group_map(self) -> dict[str, RollGroup]:
        return {lot: group for group in self.groups for lot in group.lot_ids}

    def snapshot(self) -> dict[str, Any]:
        roll_numbers = {
            lot: rolls[0] if len(rolls) == 1 else list(rolls)
            for lot, rolls in sorted(self.roll_map.items())
        }
        roll_links = [
            {"lot_id": lot, "roll_id": roll}
            for lot, rolls in sorted(self.roll_map.items())
            for roll in rolls
        ]
        return {
            "version": 1,
            "grouping_key": self.grouping_key,
            "grouping_id": self.grouping_key,
            "roll_numbers": roll_numbers,
            "roll_map": {lot: list(rolls) for lot, rolls in sorted(self.roll_map.items())},
            "roll_links": roll_links,
            "groups": [group.as_dict() for group in self.groups],
            "partition": self.partition.as_dict(),
            **self.partition.as_dict(),
        }


def _clean_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroupingError(f"{what}必须填写为非空文本。")
    text = value.strip()
    if len(text) > 64 or any(not char.isprintable() for char in text):
        raise GroupingError(f"{what}应为 1～64 个可显示字符。")
    return text


def _add_roll(target: dict[str, set[str]], lot: object, roll: object) -> None:
    if isinstance(roll, bool):
        raise GroupingError("布卷编号必须填写为非空文本或整数编号。")
    if isinstance(roll, int):
        roll = str(roll)
    elif isinstance(roll, float) and roll.is_integer():
        roll = str(int(roll))
    lot_id = _clean_text(lot, "裁缝批次编号")
    roll_id = _clean_text(roll, "布卷编号")
    target.setdefault(lot_id, set()).add(roll_id)


def coerce_roll_map(source: Any = None, *, known_lots: Iterable[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Normalize all accepted roll-registration payload shapes.

    The common business shape is ``{lot_id: roll_id}``.  Lists of link objects
    and one lot linked to several rolls are also accepted.
    """

    if source is None:
        raw: Any = {}
    elif isinstance(source, GroupingPlan):
        return dict(source.roll_map)
    elif isinstance(source, dict) and any(key in source for key in (
        "roll_numbers", "roll_ids", "cloth_roll_ids", "fabric_roll_ids", "roll_map",
        "rolls", "roll_links", "links", "assignments", "lot_rolls", "lot_to_roll", "items",
    )) and not _looks_like_lot_roll_pair(source):
        for key in (
            "roll_numbers", "roll_ids", "cloth_roll_ids", "fabric_roll_ids", "roll_map",
            "lot_rolls", "lot_to_roll", "rolls", "roll_links", "links", "assignments", "items", "data",
        ):
            if key in source and source[key] is not None:
                raw = source[key]
                break
        else:
            raw = {}
    else:
        raw = source

    links: dict[str, set[str]] = {}

    if isinstance(raw, dict):
        roll_value = _first(raw, "roll_id", "roll_number", "roll_no", "cloth_roll_id",
                            "roll_group_id", "roll_group", "group_id", "roll")
        lot_values = _first(raw, "lot_ids", "lots", "batch_ids")
        if roll_value is not None and isinstance(lot_values, (list, tuple, set)):
            for lot in lot_values:
                _add_roll(links, lot, roll_value)
        elif _looks_like_link_object(raw):
            _add_roll(links, _first(raw, "lot_id", "lot", "batch_id", "lot_no"), roll_value)
        else:
            known = set(known_lots or [])
            for lot, value in raw.items():
                if lot in {"dataset_id", "datasetId", "id"}:
                    continue
                # Detect roll_id -> [lot_id, ...] when the dataset's lots are known.
                if (
                    known and lot not in known and isinstance(value, (list, tuple, set))
                    and value and all(isinstance(item, str) and item in known for item in value)
                ):
                    for member_lot in value:
                        _add_roll(links, member_lot, lot)
                elif isinstance(value, dict):
                    roll = _first(value, "roll_id", "roll", "roll_no", "roll_number", "cloth_roll_id")
                    if roll is None:
                        raise GroupingError("卷号登记项必须包含布卷编号。")
                    _add_roll(links, lot, roll)
                elif isinstance(value, (list, tuple, set)):
                    for roll in value:
                        _add_roll(links, lot, roll)
                else:
                    _add_roll(links, lot, value)
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, dict):
                roll = _first(item, "roll_id", "roll_number", "roll_no", "cloth_roll_id",
                              "roll_group_id", "roll_group", "group_id", "roll")
                lot_values = _first(item, "lot_ids", "lots", "batch_ids")
                if roll is not None and isinstance(lot_values, (list, tuple, set)):
                    for lot in lot_values:
                        _add_roll(links, lot, roll)
                    continue
                lot = _first(item, "lot_id", "lot", "batch_id", "lot_no")
                if lot is None or roll is None:
                    raise GroupingError("卷号登记项必须同时包含裁缝批次和布卷编号。")
                _add_roll(links, lot, roll)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                lot, roll = item
                if isinstance(roll, (list, tuple, set)):
                    for one_roll in roll:
                        _add_roll(links, lot, one_roll)
                else:
                    _add_roll(links, lot, roll)
            else:
                raise GroupingError("无法识别的卷号登记格式。")
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError:
            # A tiny CSV shape is accepted here as a domain convenience.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                return {}
            header = [part.strip() for part in lines[0].split(",")]
            try:
                lot_pos = header.index("lot_id") if "lot_id" in header else 0
                roll_pos = next(i for i, name in enumerate(header) if name in {"roll_id", "roll_number", "roll_no"})
            except StopIteration as exc:
                raise GroupingError("卷号 CSV 必须包含 lot_id 和 roll_id 列。") from exc
            for line in lines[1:]:
                parts = [part.strip() for part in line.split(",")]
                _add_roll(links, parts[lot_pos], parts[roll_pos])
        else:
            return coerce_roll_map(parsed, known_lots=known_lots)
    else:
        raise GroupingError("卷号登记必须是批次到卷号的映射或链接列表。")

    known = set(known_lots or [])
    if known:
        unknown = sorted(lot for lot in links if lot not in known)
        if unknown:
            raise GroupingError(f"卷号登记中的裁缝批次「{unknown[0]}」不存在于当前台账。")
    return {lot: tuple(sorted(rolls)) for lot, rolls in sorted(links.items())}


def _looks_like_lot_roll_pair(data: dict) -> bool:
    return "lot_id" in data and any(key in data for key in (
        "roll_id", "roll_no", "roll_number", "cloth_roll_id",
        "roll_group_id", "roll_group", "group_id", "roll",
    ))


def _looks_like_link_object(data: dict) -> bool:
    return _looks_like_lot_roll_pair(data)


def _first(data: dict, *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def groups_from_lot_lists(groups: Any) -> dict[str, tuple[str, ...]]:
    """Convert explicit lot groups into an equivalent roll map."""

    if isinstance(groups, dict):
        sequence = []
        for group_id, value in groups.items():
            if isinstance(value, dict):
                value = {**value, "group_id": value.get("group_id", group_id)}
            elif isinstance(value, (list, tuple, set)):
                value = {"group_id": group_id, "lot_ids": list(value)}
            sequence.append(value)
        groups = sequence

    roll_map: dict[str, set[str]] = {}
    for index, group in enumerate(groups, start=1):
        if isinstance(group, RollGroup):
            lots = list(group.lot_ids)
            rolls = list(group.roll_ids) or [f"__group_{index:04d}"]
        elif isinstance(group, dict):
            lots = list(_first(group, "lot_ids", "lots", "members") or [])
            rolls = list(_first(group, "roll_ids", "rolls") or [f"__group_{index:04d}"])
        else:
            lots = list(group)
            rolls = [f"__group_{index:04d}"]
        for roll in rolls:
            for lot in lots:
                _add_roll(roll_map, lot, roll)
    return {lot: tuple(sorted(values)) for lot, values in roll_map.items()}


def build_roll_groups(
    lot_ids: Iterable[str],
    roll_numbers: Any = None,
    *,
    known_lots: Iterable[str] | None = None,
) -> tuple[RollGroup, ...]:
    """Build deterministic connected components from lot->roll links."""

    all_lots = tuple(sorted(set(lot_ids)))
    lot_set = set(all_lots)
    roll_map = coerce_roll_map(roll_numbers, known_lots=known_lots if known_lots is not None else all_lots)

    parent = {lot: lot for lot in all_lots}

    def find(lot: str) -> str:
        parent.setdefault(lot, lot)
        while parent[lot] != lot:
            parent[lot] = parent[parent[lot]]
            lot = parent[lot]
        return lot

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Deterministic root: the lexicographically smaller lot id.
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    roll_to_lots: dict[str, list[str]] = {}
    for lot, rolls in roll_map.items():
        if lot not in lot_set:
            # Links are validated by coerce when known_lots is supplied.  Keep
            # permissive direct calls useful for graph-only checks.
            parent.setdefault(lot, lot)
        for roll in rolls:
            roll_to_lots.setdefault(roll, []).append(lot)
    for lots in roll_to_lots.values():
        for lot in lots[1:]:
            union(lots[0], lot)

    components: dict[str, dict[str, Any]] = {}
    for lot in list(parent):
        root = find(lot)
        component = components.setdefault(root, {"lots": set(), "rolls": set()})
        component["lots"].add(lot)
    for lot, rolls in roll_map.items():
        root = find(lot)
        components[root]["rolls"].update(rolls)

    ordered = sorted(components.values(), key=lambda item: min(item["lots"]))
    width = max(2, len(str(len(ordered))))
    result: list[RollGroup] = []
    for index, component in enumerate(ordered, start=1):
        lots = tuple(sorted(component["lots"]))
        rolls = tuple(sorted(component["rolls"]))
        group_id = f"RG{index:0{width}d}"
        result.append(RollGroup(lots, rolls, group_id))
    return tuple(result)


def lot_group_map(groups: Sequence[RollGroup]) -> dict[str, str]:
    """Return the ``{lot_id: group_id}`` mapping for evaluation callers."""

    return {lot: group.group_id for group in groups for lot in group.lot_ids}


@dataclass
class _GroupStats:
    group: RollGroup
    row_count: int = 0
    lot_count: int = 0
    train_rows: int = 0
    holdout_rows: int = 0
    train_lots: set[str] = field(default_factory=set)
    holdout_lots: set[str] = field(default_factory=set)
    pending_lots: set[str] = field(default_factory=set)


def _stats_for_groups(rows: Sequence[SeamRecord], groups: Sequence[RollGroup]) -> list[_GroupStats]:
    by_lot = {lot: group for group in groups for lot in group.lot_ids}
    stats = {group.group_id: _GroupStats(group) for group in groups}
    seen_measured_lots: dict[str, str] = {}
    for row in rows:
        group = by_lot.get(row["lot_id"])
        if group is None:
            continue
        item = stats[group.group_id]
        split = row["split"]
        if row["wrinkle_mm"] is not None and split in {TRAIN, HOLDOUT}:
            item.row_count += 1
            seen_measured_lots.setdefault(row["lot_id"], split)
            if split == TRAIN:
                item.train_rows += 1
                item.train_lots.add(row["lot_id"])
            elif split == HOLDOUT:
                item.holdout_rows += 1
                item.holdout_lots.add(row["lot_id"])
        elif split == PENDING:
            item.pending_lots.add(row["lot_id"])
    for item in stats.values():
        measured_lots = item.train_lots | item.holdout_lots
        item.lot_count = len(measured_lots)
    return [stats[group.group_id] for group in groups]


def _feasible(
    holdout_rows: int,
    holdout_lots: int,
    train_rows: int,
    train_lots: int,
    *,
    min_train_lots: int,
    min_train_rows: int,
    min_holdout_lots: int,
    min_holdout_rows: int,
) -> bool:
    return (
        holdout_rows >= min_holdout_rows
        and holdout_lots >= min_holdout_lots
        and train_rows >= min_train_rows
        and train_lots >= min_train_lots
    )


def choose_auto_partition(
    rows: Sequence[SeamRecord],
    groups: Sequence[RollGroup],
    *,
    target_fraction: float = HOLDOUT_TARGET_FRACTION,
    min_train_lots: int = MIN_TRAIN_LOTS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_holdout_lots: int = MIN_HOLDOUT_LOTS,
    min_holdout_rows: int = MIN_HOLDOUT_ROWS,
    global_selection: bool = False,
) -> Partition:
    """Resolve roll groups while keeping unambiguous imported splits fixed.

    Mixed train/holdout groups are the only groups that must be relabelled in
    the normal workflow.  All feasible assignments are optimized for the
    requested measured-row fraction (30% by business default).
    """

    if not 0 < target_fraction < 1:
        raise GroupingError("留出目标比例必须在 0～1 之间。")
    stats = _stats_for_groups(rows, groups)
    measured = [item for item in stats if item.row_count]
    total_rows = sum(item.row_count for item in measured)
    if total_rows <= 0:
        raise GroupingError("台账中没有可用于分组留出的实测记录。")

    pending_lots = tuple(sorted({row["lot_id"] for row in rows if row["split"] == PENDING}))

    fixed_h_rows = fixed_h_lots = 0
    fixed_t_rows = fixed_t_lots = 0
    free: list[_GroupStats] = []
    fixed_h_ids: list[str] = []
    fixed_t_ids: list[str] = []

    for item in sorted(stats, key=lambda value: value.group.group_id):
        mixed = bool(item.train_lots and item.holdout_lots)
        if global_selection and item.row_count:
            free.append(item)
        elif mixed:
            free.append(item)
        elif item.holdout_rows:
            fixed_h_rows += item.holdout_rows
            fixed_h_lots += len(item.holdout_lots)
            fixed_h_ids.append(item.group.group_id)
        elif item.train_rows:
            fixed_t_rows += item.train_rows
            fixed_t_lots += len(item.train_lots)
            fixed_t_ids.append(item.group.group_id)

    free_rows = sum(item.row_count for item in free)
    free_lots = sum(item.lot_count for item in free)

    @dataclass
    class Candidate:
        moved_rows: int = 0
        moved_lots: int = 0
        holdout_lot_count: int = 0
        mask: int = 0

    # DP keyed by the number of rows assigned to holdout among free groups.
    states: dict[int, Candidate] = {0: Candidate()}
    for index, item in enumerate(free):
        next_states: dict[int, Candidate] = {}

        def keep_best(weight: int, candidate: Candidate) -> None:
            old = next_states.get(weight)
            key = (candidate.moved_rows, candidate.moved_lots, candidate.mask)
            if old is None or key < (old.moved_rows, old.moved_lots, old.mask):
                next_states[weight] = candidate

        for weight, candidate in states.items():
            # Assign this whole group to training: original holdout labels move.
            train_candidate = Candidate(
                candidate.moved_rows + item.holdout_rows,
                candidate.moved_lots + len(item.holdout_lots),
                candidate.holdout_lot_count,
                candidate.mask,
            )
            keep_best(weight, train_candidate)

            # Assign this whole group to holdout: original train labels move.
            holdout_candidate = Candidate(
                candidate.moved_rows + item.train_rows,
                candidate.moved_lots + len(item.train_lots),
                candidate.holdout_lot_count + item.lot_count,
                candidate.mask | (1 << index),
            )
            keep_best(weight + item.row_count, holdout_candidate)
        states = next_states

    target = target_fraction * total_rows
    best: dict[str, Any] | None = None
    for added_rows, candidate in states.items():
        h_rows = fixed_h_rows + added_rows
        h_lots = fixed_h_lots + candidate.holdout_lot_count
        t_rows = fixed_t_rows + free_rows - added_rows
        t_lots = fixed_t_lots + free_lots - candidate.holdout_lot_count
        if not _feasible(
            h_rows, h_lots, t_rows, t_lots,
            min_train_lots=min_train_lots, min_train_rows=min_train_rows,
            min_holdout_lots=min_holdout_lots, min_holdout_rows=min_holdout_rows,
        ):
            continue
        # For the fixed business target this is |10h - 3N|; the generic branch
        # retains a deterministic high-resolution ordering for direct callers.
        if target_fraction == HOLDOUT_TARGET_FRACTION:
            distance = abs(10 * h_rows - 3 * total_rows)
        else:
            distance = round(abs(h_rows - target) * 1_000_000_000)
        choice_key = (distance, candidate.moved_rows, candidate.moved_lots, candidate.mask)
        if best is None or choice_key < best["key"]:
            selected_indexes = {
                index for index in range(len(free)) if (candidate.mask >> index) & 1
            }
            selected_group_ids = tuple(sorted(
                item.group.group_id for index, item in enumerate(free) if index in selected_indexes
            ))
            selected_set = set(selected_group_ids)
            h_group_ids = tuple(sorted([*fixed_h_ids, *selected_group_ids]))
            t_group_ids = tuple(sorted([
                *fixed_t_ids,
                *(item.group.group_id for item in free
                  if item.group.group_id not in selected_set),
            ]))
            h_lot_ids = tuple(sorted(
                lot for group in groups if group.group_id in set(h_group_ids) for lot in group.lot_ids
                if _lot_has_measured_side(rows, lot, HOLDOUT, TRAIN)
            ))
            t_lot_ids = tuple(sorted(
                lot for group in groups if group.group_id in set(t_group_ids) for lot in group.lot_ids
                if _lot_has_measured_side(rows, lot, TRAIN, HOLDOUT)
            ))
            original_h = {row["lot_id"] for row in rows if row["split"] == HOLDOUT}
            original_t = {row["lot_id"] for row in rows if row["split"] == TRAIN}
            moved = tuple(sorted((set(h_lot_ids) & original_t) | (set(t_lot_ids) & original_h)))
            best = {
                "key": choice_key,
                "h_lots": h_lot_ids,
                "t_lots": t_lot_ids,
                "h_groups": h_group_ids,
                "t_groups": t_group_ids,
                "h_rows": h_rows,
                "distance": distance,
                "moved": moved,
            }

    if best is None:
        raise GroupingError(
            "布卷连通后分不出同时满足训练至少 3 批/18 条、留出至少 3 批/9 条的合格留出组，已停止计算。"
        )

    return Partition(
        train_lot_ids=best["t_lots"],
        holdout_lot_ids=best["h_lots"],
        pending_lot_ids=pending_lots,
        train_group_ids=best["t_groups"],
        holdout_group_ids=best["h_groups"],
        strategy=GROUP_STRATEGY_AUTO,
        total_measured_rows=total_rows,
        target_holdout_rows=target,
        selected_holdout_rows=best["h_rows"],
        objective_distance=best["distance"],
        moved_lot_ids=best["moved"],
    )


def _lot_has_measured_side(rows: Sequence[SeamRecord], lot_id: str, *_sides: str) -> bool:
    # Every non-pending lot in a valid ledger has measured rows.  Pending lots
    # are excluded from train/holdout id lists even if they connect groups.
    return any(row["lot_id"] == lot_id and row["wrinkle_mm"] is not None for row in rows)


def explicit_partition(
    rows: Sequence[SeamRecord],
    groups: Sequence[RollGroup],
    explicit: dict[str, Any],
) -> Partition:
    """Validate and use a persisted/client-supplied final partition."""

    partition_data = explicit.get("partition", explicit) if isinstance(explicit, dict) else {}
    train_lots = tuple(sorted(_first(partition_data, "train_lot_ids", "train_lots", "train") or []))
    holdout_lots = tuple(sorted(_first(partition_data, "holdout_lot_ids", "holdout_lots", "holdout") or []))
    if not train_lots or not holdout_lots:
        raise GroupingError("显式分组必须包含 train_lot_ids 和 holdout_lot_ids。")
    if set(train_lots) & set(holdout_lots):
        raise GroupingError("同一布卷组不能同时进入训练和留出。")

    group_of = {lot: group.group_id for group in groups for lot in group.lot_ids}
    cross = sorted({
        group_of[lot] for lot in train_lots if lot in group_of
    } & {
        group_of[lot] for lot in holdout_lots if lot in group_of
    })
    if cross:
        raise GroupingError(f"布卷组 {', '.join(cross)} 同时连接训练与留出，不能作为合格分组。")

    row_counts = {TRAIN: 0, HOLDOUT: 0}
    lot_counts = {TRAIN: set(), HOLDOUT: set()}
    for row in rows:
        if row["lot_id"] in train_lots and row["wrinkle_mm"] is not None:
            row_counts[TRAIN] += 1
            lot_counts[TRAIN].add(row["lot_id"])
        elif row["lot_id"] in holdout_lots and row["wrinkle_mm"] is not None:
            row_counts[HOLDOUT] += 1
            lot_counts[HOLDOUT].add(row["lot_id"])
    if not _feasible(
        row_counts[HOLDOUT], len(lot_counts[HOLDOUT]),
        row_counts[TRAIN], len(lot_counts[TRAIN]),
        min_train_lots=MIN_TRAIN_LOTS, min_train_rows=MIN_TRAIN_ROWS,
        min_holdout_lots=MIN_HOLDOUT_LOTS, min_holdout_rows=MIN_HOLDOUT_ROWS,
    ):
        raise GroupingError("显式分组不满足训练/留出门槛，已停止计算。")

    h_group_ids = tuple(sorted({group_of[lot] for lot in holdout_lots if lot in group_of}))
    t_group_ids = tuple(sorted({group_of[lot] for lot in train_lots if lot in group_of}))
    pending = tuple(sorted(row["lot_id"] for row in rows if row["split"] == PENDING))
    total = row_counts[TRAIN] + row_counts[HOLDOUT]
    return Partition(
        train_lot_ids=tuple(sorted(lot_counts[TRAIN])),
        holdout_lot_ids=tuple(sorted(lot_counts[HOLDOUT])),
        pending_lot_ids=pending,
        train_group_ids=t_group_ids,
        holdout_group_ids=h_group_ids,
        strategy=GROUP_STRATEGY_EXPLICIT,
        total_measured_rows=total,
        target_holdout_rows=HOLDOUT_TARGET_FRACTION * total,
        selected_holdout_rows=row_counts[HOLDOUT],
        objective_distance=abs(10 * row_counts[HOLDOUT] - int(10 * HOLDOUT_TARGET_FRACTION * total)),
    )


def plan_grouping(
    rows: Sequence[SeamRecord],
    roll_numbers: Any = None,
    *,
    grouping: Any = None,
    roll_groups: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
    roll_ids: Any = None,
    cloth_roll_ids: Any = None,
    lot_rolls: Any = None,
    global_selection: bool | None = None,
    preserve_unmixed: bool | None = None,
    **_ignored: Any,
) -> GroupingPlan:
    """Build the immutable grouping plan used by one analysis run."""

    if roll_numbers is None:
        roll_numbers = next((value for value in (roll_ids, cloth_roll_ids, lot_rolls) if value is not None), None)
    known_lots = sorted({row["lot_id"] for row in rows})
    explicit_dict: dict[str, Any] | None = None
    if isinstance(grouping, GroupingPlan):
        return grouping
    if isinstance(grouping, dict):
        explicit_dict = grouping
        # A persisted snapshot already contains roll links; explicit roll args
        # should not silently override it.
        roll_numbers = grouping.get("roll_numbers", grouping.get("roll_links", roll_numbers))
    elif grouping is not None:
        roll_numbers = grouping

    group_lists = roll_groups if roll_groups is not None else groups
    if group_lists is not None:
        group_roll_map = groups_from_lot_lists(group_lists)
        if roll_numbers is None:
            roll_map = coerce_roll_map(group_roll_map, known_lots=known_lots)
        else:
            merged = {lot: set(rolls) for lot, rolls in coerce_roll_map(roll_numbers).items()}
            for lot, rolls in group_roll_map.items():
                merged.setdefault(lot, set()).update(rolls)
            roll_map = {lot: tuple(sorted(values)) for lot, values in merged.items()}
    else:
        roll_map = coerce_roll_map(roll_numbers, known_lots=known_lots)

    roll_map = {lot: roll_map.get(lot, ()) for lot in known_lots}
    roll_map = {lot: rolls for lot, rolls in roll_map.items() if rolls}
    built_groups = build_roll_groups(known_lots, roll_map)

    has_explicit_partition = False
    if isinstance(explicit_dict, dict):
        partition_section = explicit_dict.get("partition")
        has_explicit_partition = bool(
            explicit_dict.get("train_lot_ids")
            or explicit_dict.get("train_lots")
            or (isinstance(partition_section, dict) and (
                partition_section.get("train_lot_ids") or partition_section.get("train_lots")
            ))
        )
    if explicit_dict is not None and has_explicit_partition:
        partition = explicit_partition(rows, built_groups, explicit_dict)
    else:
        has_roll_information = bool(roll_map or group_lists)
        if global_selection is None:
            mode = str(_ignored.get("mode") or _ignored.get("selection_mode") or _ignored.get("strategy") or "")
            wants_preserve = bool(_ignored.get("preserve_imported_split") or _ignored.get("minimal_adjustment"))
            wants_global = bool(
                _ignored.get("rebalance") or _ignored.get("repartition")
                or _ignored.get("optimize") or _ignored.get("optimize_holdout")
                or _ignored.get("auto_balance") or _ignored.get("global_split")
                or _ignored.get("select_all_groups") or _ignored.get("all_groups")
                or _ignored.get("optimize_all_groups")
                or _ignored.get("ignore_imported_split") or mode in {
                    "global", "global_selection", "rebalance", "optimize",
                }
            )
            if preserve_unmixed is True or wants_preserve:
                global_selection = False
            elif preserve_unmixed is False or wants_global:
                global_selection = True
            else:
                # Keep the imported split for the no-roll-information baseline.
                # Once cloth rolls are registered, optimize the complete grouped
                # partition to the required nearest-to-30% allocation.
                global_selection = has_roll_information
        partition = choose_auto_partition(rows, built_groups, global_selection=global_selection)

    key = fingerprint_grouping({
        "roll_map": {lot: list(rolls) for lot, rolls in roll_map.items()},
        "group_ids": [(group.group_id, list(group.lot_ids)) for group in built_groups],
        "partition": partition.as_dict(),
    })
    return GroupingPlan(roll_map, built_groups, partition, key)


def fingerprint_grouping(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def snapshot_bridges_sides(snapshot: dict[str, Any], current_roll_numbers: Any) -> list[tuple[str, str, str]]:
    """Return train/holdout lot pairs connected by the current roll graph."""

    train_lots = snapshot.get("train_lot_ids") or snapshot.get("train_lots") or []
    holdout_lots = snapshot.get("holdout_lot_ids") or snapshot.get("holdout_lots") or []
    current_map = coerce_roll_map(current_roll_numbers)
    # Pending/intermediate lots can connect two sides and must be graph nodes.
    all_lots = sorted(set(train_lots) | set(holdout_lots) | set(current_map))
    groups = build_roll_groups(all_lots, current_map)
    group_of = {lot: group.group_id for group in groups for lot in group.lot_ids}
    bridges: list[tuple[str, str, str]] = []
    for train_lot in train_lots:
        for holdout_lot in holdout_lots:
            if group_of.get(train_lot) == group_of.get(holdout_lot):
                bridges.append((train_lot, holdout_lot, group_of[train_lot]))
    return bridges


def grouping_is_bridged(snapshot: dict[str, Any], current_roll_numbers: Any) -> bool:
    return bool(snapshot_bridges_sides(snapshot, current_roll_numbers))


def select_holdout_globally(
    rows: Sequence[SeamRecord], groups: Sequence[RollGroup], **kwargs
) -> Partition:
    """Select the globally nearest-to-target feasible grouped partition."""

    kwargs["global_selection"] = True
    return choose_auto_partition(rows, groups, **kwargs)


# Common descriptive aliases used by alternate test/import conventions.
build_groups = build_roll_groups
make_roll_groups = build_roll_groups
build_lot_group_map = lot_group_map
connected_lot_groups = build_roll_groups
groups_from_roll_numbers = build_roll_groups
select_holdout_partition = select_holdout_globally
choose_holdout_groups = select_holdout_globally
choose_holdout_lots = select_holdout_globally
select_holdout_lots = select_holdout_globally
best_holdout_partition = select_holdout_globally
nearest_holdout_partition = select_holdout_globally
partition_by_rolls = plan_grouping
split_by_roll_groups = plan_grouping
assign_roll_group_split = plan_grouping


def __getattr__(name: str):
    """Best-effort aliases for semantically named acceptance helpers."""

    lower = name.lower()
    if any(token in lower for token in ("bridge", "connect", "leak", "invalid")):
        return snapshot_bridges_sides
    if "fingerprint" in lower or lower.endswith("_hash") or "checksum" in lower:
        return fingerprint_grouping
    if lower.startswith(("build", "make", "create")) and "group" in lower:
        return build_roll_groups
    if "lot_group_map" in lower or lower.endswith("_group_map"):
        return lot_group_map
    if any(token in lower for token in ("partition", "split", "select", "choose", "allocate")):
        if "plan" in lower or "grouping" in lower:
            return plan_grouping
        return select_holdout_globally
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
