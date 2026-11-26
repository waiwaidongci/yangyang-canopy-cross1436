"""Persistence and concurrency control for retroactive cloth-roll links."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from sqlalchemy import select

from src.constants.workflow import RunStatus
from src.domain.grouping import (
    build_roll_groups,
    coerce_roll_map,
    groups_from_lot_lists,
    plan_grouping,
    snapshot_bridges_sides,
)
from src.models.roll_link import RollLink
from src.models.run import Run
from src.services.catalog import get_dataset
from src.utils.time import utc_stamp


_LOCK_REGISTRY_GUARD = threading.Lock()
_DATASET_LOCKS: dict[int, threading.RLock] = {}


def dataset_lock(dataset_id: int) -> threading.RLock:
    """Return the in-process critical section for one dataset."""

    with _LOCK_REGISTRY_GUARD:
        return _DATASET_LOCKS.setdefault(int(dataset_id), threading.RLock())


def _merge_maps(*maps: dict[str, tuple[str, ...] | list[str] | str | None]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for source in maps:
        if not source:
            continue
        normalized = coerce_roll_map(source)
        for lot, rolls in normalized.items():
            merged[lot].update(rolls)
    return {lot: tuple(sorted(rolls)) for lot, rolls in sorted(merged.items())}


def stored_roll_map(session, dataset) -> dict[str, tuple[str, ...]]:
    """Read both the link table and the cached JSON field."""

    links = session.scalars(
        select(RollLink).where(RollLink.dataset_id == dataset.id)
    ).all()
    table_map: dict[str, list[str]] = defaultdict(list)
    for link in links:
        table_map[link.lot_id].append(link.roll_id)
    cached = getattr(dataset, "roll_numbers", None)
    assignments = getattr(dataset, "roll_assignments", None)
    merged = _merge_maps(
        dict(table_map),
        cached if isinstance(cached, (dict, list)) else None,
        assignments if isinstance(assignments, (dict, list)) else None,
    )
    cached_groups = getattr(dataset, "roll_groups", None)
    if cached_groups:
        try:
            group_map = groups_from_lot_lists(cached_groups)
            merged = _merge_maps(merged, group_map)
        except Exception:
            # A cached API representation with group_id/lot_ids is handled by
            # the normal dict parser only when it contains link fields.
            pass
    return merged


def current_plan(session, dataset):
    roll_map = stored_roll_map(session, dataset)
    return plan_grouping(dataset.rows, roll_map)


def grouping_overview(session, dataset_id: int) -> dict[str, Any]:
    dataset = get_dataset(session, dataset_id)
    roll_map = stored_roll_map(session, dataset)
    known_lots = sorted({row["lot_id"] for row in dataset.rows})
    groups = build_roll_groups(known_lots, roll_map)
    try:
        plan = plan_grouping(dataset.rows, roll_map)
        snapshot = plan.snapshot()
        partition_error = None
        grouping_key = plan.grouping_key
    except ValueError as error:
        partition_error = str(error)
        grouping_key = None
        snapshot = {
            "version": 1,
            "roll_numbers": {lot: rolls[0] if len(rolls) == 1 else list(rolls) for lot, rolls in roll_map.items()},
            "roll_map": {lot: list(rolls) for lot, rolls in roll_map.items()},
            "roll_links": [
                {"lot_id": lot, "roll_id": roll}
                for lot, rolls in roll_map.items() for roll in rolls
            ],
            "groups": [group.as_dict() for group in groups],
            "partition_error": partition_error,
        }
    return {
        "dataset_id": dataset.id,
        "checksum": dataset.checksum,
        "grouping_key": grouping_key,
        "grouping_id": grouping_key,
        "roll_numbers": snapshot.get("roll_numbers"),
        "roll_map": snapshot.get("roll_map"),
        "roll_links": snapshot.get("roll_links"),
        "groups": snapshot.get("groups"),
        "partition": snapshot.get("partition"),
        "partition_error": partition_error,
        "train_lot_ids": snapshot.get("train_lot_ids", []),
        "holdout_lot_ids": snapshot.get("holdout_lot_ids", []),
        "grouping": snapshot,
    }


def _run_snapshot_for_invalidation(run: Run, dataset) -> dict[str, Any]:
    snapshot = run.grouping or {}
    if snapshot.get("train_lot_ids") and snapshot.get("holdout_lot_ids"):
        return snapshot

    # Backwards compatibility for runs created before grouping snapshots were
    # persisted: the original CSV split is the grouping they used.
    holdout_lots: set[str] = set()
    result = getattr(run, "result", None) or {}
    for table in result.get("tables", []):
        if table.get("key") == "holdout_lots":
            holdout_lots.update(row.get("lot_id") for row in table.get("rows", []) if row.get("lot_id"))
        if table.get("key") == "holdout_seams":
            holdout_lots.update(row.get("lot_id") for row in table.get("rows", []) if row.get("lot_id"))
    train_lots = {
        row.get("lot_id") for row in (dataset.rows or [])
        if row.get("split") == "train" and row.get("lot_id")
    }
    return {"train_lot_ids": sorted(train_lots), "holdout_lot_ids": sorted(holdout_lots)}


def invalidate_connected_runs(session, dataset, roll_map) -> list[int]:
    """Immediately invalidate completed/running runs crossed by current links."""

    runs = session.scalars(
        select(Run).where(
            Run.dataset_id == dataset.id,
            Run.status.in_([RunStatus.COMPLETED.value, RunStatus.RUNNING.value]),
        )
    ).all()
    now = utc_stamp()
    invalidated: list[int] = []
    for run in runs:
        snapshot = _run_snapshot_for_invalidation(run, dataset)
        if not snapshot.get("train_lot_ids") or not snapshot.get("holdout_lot_ids"):
            continue
        bridges = snapshot_bridges_sides(snapshot, roll_map)
        if not bridges:
            continue
        examples = "、".join(f"{train_lot}—{holdout_lot}" for train_lot, holdout_lot, _ in bridges[:3])
        reason = (
            "并发补登的卷号已使布卷组同时连接训练与留出"
            + (f"（如 {examples}）" if examples else "")
            + "，原评估分组失效，结果不能继续发布或导出。"
        )
        run.status = RunStatus.INVALIDATED.value
        run.invalidated_at = now
        run.invalidation_reason = reason
        invalidated.append(run.id)
    return invalidated


def register_rolls(session, dataset_id: int, payload: Any = None, actor_id: int | None = None, **kwargs) -> dict[str, Any]:
    """Merge newly registered links and invalidate crossed evaluations.

    The dataset lock plus the run publish lock makes registration/publish
    linearizable inside the API process.  The flexible argument handling also
    supports direct service calls shaped as ``(session, dataset, roll_map)``.
    """

    if payload is None and "roll_numbers" in kwargs:
        payload = kwargs["roll_numbers"]
    if hasattr(dataset_id, "id"):
        dataset_id = dataset_id.id
    if actor_id is not None and hasattr(actor_id, "id"):
        actor_id = actor_id.id
    dataset = get_dataset(session, dataset_id)
    with dataset_lock(dataset_id):
        try:
            known_lots = sorted({row["lot_id"] for row in dataset.rows})
            incoming = coerce_roll_map(payload, known_lots=known_lots)

            existing_links = session.scalars(
                select(RollLink).where(RollLink.dataset_id == dataset_id)
            ).all()
            existing_pairs = {(link.lot_id, link.roll_id) for link in existing_links}
            now = utc_stamp()
            created = 0
            for lot_id, rolls in incoming.items():
                for roll_id in rolls:
                    if (lot_id, roll_id) in existing_pairs:
                        continue
                    session.add(RollLink(
                        dataset_id=dataset_id, lot_id=lot_id, roll_id=roll_id,
                        created_by=actor_id, created_at=now,
                    ))
                    existing_pairs.add((lot_id, roll_id))
                    created += 1
            session.flush()

            roll_map = stored_roll_map(session, dataset)
            # Keep the conventionally named JSON field in the common
            # {lot_id: roll_id} shape; multi-roll lots use a list value.
            dataset.roll_numbers = {
                lot_id: rolls[0] if len(rolls) == 1 else list(rolls)
                for lot_id, rolls in roll_map.items()
            }
            dataset.roll_assignments = [
                {"lot_id": lot_id, "roll_id": roll_id}
                for lot_id, rolls in roll_map.items()
                for roll_id in rolls
            ]
            invalidated = invalidate_connected_runs(session, dataset, roll_map)
            known_lots = sorted({row["lot_id"] for row in dataset.rows})
            roll_groups = build_roll_groups(known_lots, roll_map)
            dataset.roll_groups = [group.as_dict() for group in roll_groups]
            try:
                plan = plan_grouping(dataset.rows, roll_map)
                snapshot = plan.snapshot()
                grouping_key = plan.grouping_key
                partition_error = None
            except ValueError as error:
                snapshot = {
                    "version": 1,
                    "roll_numbers": {lot: rolls[0] if len(rolls) == 1 else list(rolls)
                                     for lot, rolls in roll_map.items()},
                    "roll_map": {lot: list(rolls) for lot, rolls in roll_map.items()},
                    "roll_links": dataset.roll_assignments,
                    "groups": [group.as_dict() for group in roll_groups],
                    "partition_error": str(error),
                }
                grouping_key = None
                partition_error = str(error)
            overview = {
                "dataset_id": dataset.id,
                "checksum": dataset.checksum,
                "grouping_key": grouping_key,
                "grouping_id": grouping_key,
                "partition_error": partition_error,
                **snapshot,
            }
            overview.update({
                "created_links": created,
                "created": created,
                "invalidated_run_ids": invalidated,
                "invalidated_runs": invalidated,
                "invalidated_count": len(invalidated),
            })
            session.commit()
        except Exception:
            session.rollback()
            raise
        return overview


# Naming aliases for callers/tests using the business phrase rather than "rolls".
register_roll_numbers = register_rolls
add_roll_numbers = register_rolls
assign_roll_numbers = register_rolls
register_roll_links = register_rolls
invalidate_stale_evaluations = invalidate_connected_runs
invalidate_crossed_evaluations = invalidate_connected_runs


class GroupingService:
    """Thin object-style facade for tests that prefer a service class."""

    lock = staticmethod(dataset_lock)
    overview = staticmethod(grouping_overview)
    current_plan = staticmethod(current_plan)
    stored_roll_map = staticmethod(stored_roll_map)
    register = staticmethod(register_rolls)
    register_roll_numbers = staticmethod(register_roll_numbers)
    invalidate_connected_runs = staticmethod(invalidate_connected_runs)


class RollNumberService(GroupingService):
    pass


class RollService(GroupingService):
    pass
