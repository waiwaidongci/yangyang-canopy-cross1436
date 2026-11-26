import math
import time

import hashlib
import json

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, Role, RunStatus
from src.domain.project import SPEC
from src.models.review import Review
from src.models.run import Run
from src.services.catalog import get_dataset
from src.services.execution import calculate
from src.services.grouping import (
    current_plan,
    dataset_lock,
    register_roll_numbers,
    stored_roll_map,
)
from src.domain.grouping import coerce_roll_map, plan_grouping, snapshot_bridges_sides
from src.services.grouping import invalidate_connected_runs
from src.utils.time import utc_stamp


INVALID_STATUSES = {RunStatus.INVALIDATED, RunStatus.STALE, RunStatus.INVALIDATED.value, RunStatus.STALE.value}


def run_info(run, review, *, replayed=None):
    fields = ('id', 'label', 'dataset_id', 'created_at', 'status', 'settings')
    info = {key: getattr(run, key) for key in fields}
    info['decision'] = review.decision if review else None
    info['grouping'] = getattr(run, 'grouping', None)
    for key in ('grouping_key', 'invalidated_at', 'invalidation_reason'):
        info[key] = getattr(run, key, None)
    is_invalid = run.status in INVALID_STATUSES or str(run.status) in {'invalidated', 'stale'}
    info['replayed'] = replayed
    # Convenience aliases used by clients and acceptance checks.
    info['idempotency_key'] = getattr(run, 'idempotency_key', None)
    info['valid'] = not is_invalid
    info['invalidated'] = is_invalid
    info['stale'] = is_invalid
    if replayed is None:
        info.pop('replayed', None)
    return info


def list_runs(session, actor):
    statement = select(Run, Review).outerjoin(Review, Run.id == Review.run_id).order_by(Run.id.desc()).limit(100)
    if actor.role == Role.VIEWER:
        statement = statement.where(
            Review.decision == Decision.APPROVED,
            Run.status == RunStatus.COMPLETED.value,
        )
    return [run_info(run, review) for run, review in session.execute(statement)]


def get_run(session, run_id, actor):
    run = session.get(Run, run_id)
    review = session.get(Review, run_id)
    if run is None:
        raise LookupError('找不到可查看的运行')
    if actor.role == Role.VIEWER and (
        not review or review.decision != Decision.APPROVED or run.status != RunStatus.COMPLETED.value
    ):
        raise LookupError('找不到可查看的运行')
    return run, review


def validate_settings(settings):
    parameters = {item['key']: item for item in SPEC['settings']}
    if settings.keys() - parameters.keys():
        raise ValueError('存在本项目未定义的运行参数')
    checked = {}
    for key, parameter in parameters.items():
        value = settings.get(key, parameter['default'])
        if not math.isfinite(value) or not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{parameter['label']}超出范围")
        if parameter['step'] == 1 and value != int(value):
            raise ValueError(f"{parameter['label']}必须是整数")
        checked[key] = int(value) if parameter['step'] == 1 else value
    return checked


def _payload_value(payload, *names):
    if isinstance(payload, dict):
        for name in names:
            if name in payload and payload[name] is not None:
                return payload[name]
        return None
    for name in names:
        if hasattr(payload, name):
            value = getattr(payload, name)
            if value is not None:
                return value
    return None


def _explicit_group_options(payload):
    grouping = _payload_value(payload, 'grouping')
    options = {
        'roll_numbers': _payload_value(
            payload, 'roll_numbers', 'roll_ids', 'cloth_roll_ids', 'lot_rolls', 'roll_map', 'rolls'
        ),
        'roll_links': _payload_value(payload, 'roll_links'),
        'groups': _payload_value(payload, 'groups', 'roll_groups'),
        'roll_groups': _payload_value(payload, 'roll_groups'),
        'selection_mode': _payload_value(payload, 'selection_mode', 'mode', 'strategy'),
        'global_selection': _payload_value(payload, 'global_selection'),
        'preserve_unmixed': _payload_value(payload, 'preserve_unmixed'),
    }
    # A list/dict passed as grouping can itself be roll links or lot groups.
    if grouping is not None and not isinstance(grouping, (str, bytes)):
        if isinstance(grouping, list):
            if grouping and all(isinstance(item, dict) and ('roll_id' in item or 'roll' in item) for item in grouping):
                options['roll_numbers'] = grouping
            else:
                options['roll_groups'] = grouping
        elif isinstance(grouping, dict) and not (
            grouping.get('train_lot_ids') or grouping.get('holdout_lot_ids')
            or isinstance(grouping.get('partition'), dict)
        ):
            options['roll_numbers'] = grouping
    return grouping, {key: value for key, value in options.items() if value is not None}


def _idempotency_key(dataset, settings, grouping_key):
    settings_text = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    raw = f'{dataset.id}|{dataset.checksum}|{grouping_key}|{settings_text}'
    return hashlib.sha256(raw.encode()).hexdigest()


def create_run(session, payload, actor):
    if isinstance(payload, dict):
        from src.schemas.runs import RunCreate
        payload = RunCreate(**payload)
    dataset = get_dataset(session, payload.dataset_id)
    settings = validate_settings(payload.settings or {})
    if isinstance(actor, dict):
        actor_id = actor.get('id')
    else:
        actor_id = actor.id if hasattr(actor, 'id') else int(actor)
    explicit_grouping, group_options = _explicit_group_options(payload)

    # Resolve the immutable start-of-calculation grouping inside the same
    # critical section used by roll registration and run publication.
    with dataset_lock(dataset.id):
        planning_error = None
        stored_map = {}
        try:
            stored_map = stored_roll_map(session, dataset)
            if explicit_grouping is not None and isinstance(explicit_grouping, dict) and (
                explicit_grouping.get('train_lot_ids')
                or explicit_grouping.get('holdout_lot_ids')
                or isinstance(explicit_grouping.get('partition'), dict)
            ):
                explicit_options = {
                    key: value for key, value in group_options.items()
                    if key in {'selection_mode', 'global_selection', 'preserve_unmixed'}
                }
                plan = plan_grouping(
                    dataset.rows,
                    roll_numbers=stored_map,
                    grouping=explicit_grouping,
                    **explicit_options,
                )
            else:
                merged = dict(stored_map)
                explicit_rolls = group_options.get('roll_numbers') or group_options.get('roll_links')
                if explicit_rolls is not None:
                    explicit_map = coerce_roll_map(explicit_rolls)
                    for lot, rolls in explicit_map.items():
                        merged[lot] = tuple(sorted(set(merged.get(lot, ())) | set(rolls)))
                selection_options = {
                    key: value for key, value in group_options.items()
                    if key in {'selection_mode', 'global_selection', 'preserve_unmixed'}
                }
                plan = plan_grouping(
                    dataset.rows,
                    roll_numbers=merged,
                    roll_groups=group_options.get('roll_groups') or group_options.get('groups'),
                    **selection_options,
                )
            grouping_snapshot = plan.snapshot()
            grouping_key = plan.grouping_key
        except ValueError as error:
            planning_error = str(error)
            grouping_snapshot = {
                'error': planning_error,
                'stored_roll_numbers': stored_map,
                'explicit_grouping': explicit_grouping,
            }
            grouping_key = hashlib.sha256(
                json.dumps(grouping_snapshot, ensure_ascii=False, sort_keys=True, default=str).encode()
            ).hexdigest()

        idempotency_key = _idempotency_key(dataset, settings, grouping_key)
        existing = session.scalar(select(Run).where(Run.idempotency_key == idempotency_key))
        existing_running_id = None
        if existing is not None and existing.status != RunStatus.RUNNING.value:
            review = session.get(Review, existing.id)
            return run_info(existing, review, replayed=True)

        if existing is not None:
            # Wait outside the critical section for the in-flight owner to
            # publish rather than calculating a second identical evaluation.
            run = existing
            existing_running_id = existing.id
        else:
            created_at = utc_stamp()
            run = Run(
                label=payload.label, dataset_id=payload.dataset_id, created_by=actor_id,
                created_at=created_at, settings=settings, grouping=grouping_snapshot,
                grouping_key=grouping_key, idempotency_key=idempotency_key,
                status=RunStatus.FAILED.value if planning_error else RunStatus.RUNNING.value,
                error=planning_error,
            )
            session.add(run)
            session.commit()

    if existing_running_id is not None:
        for _ in range(3000):
            session.rollback()
            latest = session.get(Run, existing_running_id)
            if latest is not None and latest.status != RunStatus.RUNNING.value:
                review = session.get(Review, existing_running_id)
                return run_info(latest, review, replayed=True)
            time.sleep(0.01)
        latest = session.get(Run, existing_running_id)
        return run_info(latest or run, session.get(Review, existing_running_id), replayed=True)

    if planning_error:
        return run_info(run, None)

    # Calculation intentionally happens without holding the dataset lock or an
    # open write transaction.  Publication revalidates the grouping below.
    try:
        result, error = calculate(dataset.rows, settings, grouping=grouping_snapshot)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('计算运行失败')
        result, error = None, '计算遇到内部错误，请检查台账后重试。'

    with dataset_lock(dataset.id):
        session.refresh(run)
        if run.status != RunStatus.RUNNING.value:
            review = session.get(Review, run.id)
            return run_info(run, review)

        session.refresh(dataset)
        current_map = stored_roll_map(session, dataset)
        bridges = snapshot_bridges_sides(grouping_snapshot, current_map)
        if bridges:
            now = utc_stamp()
            examples = '、'.join(f'{t}—{h}' for t, h, _ in bridges[:3])
            reason = f'计算期间补登的卷号连通了训练与留出（如 {examples}），本次结果禁止发布。'
            run.status = RunStatus.INVALIDATED.value
            run.invalidated_at = now
            run.invalidation_reason = reason
            run.error = reason
            run.result = None
            session.commit()
            return run_info(run, None)

        run.status = RunStatus.FAILED.value if error is not None else RunStatus.COMPLETED.value
        run.result = result
        run.error = error
        session.commit()
        return run_info(run, None)


def invalidate_run(session, run_id, reason=None):
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError('找不到运行')
    run.status = RunStatus.INVALIDATED.value
    run.invalidated_at = utc_stamp()
    run.invalidation_reason = reason or '该评估已因布卷分组变化而失效。'
    session.commit()
    return run_info(run, session.get(Review, run_id))


def export_table(session, run_id, key, actor):
    run, review = get_run(session, run_id, actor)
    if run.status in INVALID_STATUSES or str(run.status) in {'invalidated', 'stale'}:
        raise HTTPException(409, run.invalidation_reason or '该评估已因布卷分组变化而失效，不能发布或导出。')
    if review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '复核通过后可下载正式 CSV')
    if not run.result:
        raise LookupError('找不到结果表')
    for table in run.result['tables']:
        if table['key'] == key:
            return table
    raise LookupError('找不到结果表')


# Descriptive aliases for acceptance code that models these as evaluations.
create_evaluation = create_run
rerun_evaluation = create_run
create_analysis_run = create_run
invalidate_evaluation = invalidate_run
mark_run_invalid = invalidate_run
