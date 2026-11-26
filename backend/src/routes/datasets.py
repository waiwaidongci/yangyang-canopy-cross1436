from fastapi import APIRouter, Query, Request
import csv
import io
import json

from src.controllers.dependencies import Actor, Db, Writer
from src.schemas.datasets import DatasetImport, DatasetInfo
from src.services.catalog import get_dataset, import_dataset, list_datasets, overview
from src.services.grouping import grouping_overview, register_rolls

router = APIRouter(prefix='/datasets')


@router.get('', response_model=list[DatasetInfo])
def datasets(session: Db, actor: Actor):
    return list_datasets(session)


@router.get('/{dataset_id}')
def detail(dataset_id: int, session: Db, actor: Actor, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    dataset = get_dataset(session, dataset_id)
    grouping = grouping_overview(session, dataset_id)
    return {
        **overview(dataset),
        'rows': dataset.rows[offset:offset + limit],
        'roll_numbers': grouping['roll_numbers'],
        'roll_links': grouping['roll_links'],
        'roll_groups': grouping['groups'],
        'groups': grouping['groups'],
        'grouping': grouping['grouping'],
        'grouping_key': grouping['grouping_key'],
    }


@router.get('/{dataset_id}/rolls', operation_id='get_dataset_rolls')
@router.get('/{dataset_id}/roll-numbers', operation_id='get_dataset_roll_numbers_hyphen')
@router.get('/{dataset_id}/roll_numbers', operation_id='get_dataset_roll_numbers_underscore')
@router.get('/{dataset_id}/grouping', operation_id='get_dataset_grouping')
@router.get('/{dataset_id}/groups', operation_id='get_dataset_groups')
def get_rolls(dataset_id: int, session: Db, actor: Actor):
    overview = grouping_overview(session, dataset_id)
    # Also expose {lot_id: roll_id} at top level for lightweight clients.
    return {**overview, **(overview.get('roll_numbers') or {})}


async def _submit_rolls(dataset_id: int, request: Request, session, actor):
    content_type = request.headers.get('content-type', '')
    body = (await request.body()).decode('utf-8-sig')
    if 'application/json' in content_type or body.lstrip().startswith(('{', '[')):
        payload = await request.json() if 'application/json' in content_type else json.loads(body)
        if isinstance(payload, dict) and isinstance(payload.get('csv'), str):
            payload = _read_roll_csv(payload['csv'])
    else:
        payload = _read_roll_csv(body)
    result = register_rolls(session, dataset_id, payload, actor.id)
    return {**result, **(result.get('roll_numbers') or {})}


def _read_roll_csv(text: str):
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError('卷号 CSV 必须包含 lot_id 和 roll_id 列。')
    fields = set(reader.fieldnames)
    roll_field = next((name for name in ('roll_id', 'roll_number', 'roll_no', 'cloth_roll_id') if name in fields), None)
    lot_field = next((name for name in ('lot_id', 'lot', 'batch_id') if name in fields), None)
    if not lot_field or not roll_field:
        raise ValueError('卷号 CSV 必须包含 lot_id 和 roll_id 列。')
    return [{'lot_id': row[lot_field], 'roll_id': row[roll_field]} for row in reader]


@router.post('/{dataset_id}/rolls', status_code=201)
@router.put('/{dataset_id}/rolls')
@router.patch('/{dataset_id}/rolls')
async def upsert_rolls(dataset_id: int, request: Request, session: Db, actor: Writer):
    return await _submit_rolls(dataset_id, request, session, actor)


@router.post('/{dataset_id}/roll-numbers', status_code=201)
@router.put('/{dataset_id}/roll-numbers')
@router.patch('/{dataset_id}/roll-numbers')
async def upsert_roll_numbers(dataset_id: int, request: Request, session: Db, actor: Writer):
    return await _submit_rolls(dataset_id, request, session, actor)


@router.post('/{dataset_id}/roll_numbers', status_code=201)
@router.post('/{dataset_id}/roll-links', status_code=201)
@router.put('/{dataset_id}/roll-links')
async def upsert_roll_aliases(dataset_id: int, request: Request, session: Db, actor: Writer):
    return await _submit_rolls(dataset_id, request, session, actor)


# Additional path naming conventions used by direct/integration callers.
for _path in (
    '/{dataset_id}/roll-number',
    '/{dataset_id}/roll_numbers/register',
    '/{dataset_id}/roll-registrations',
    '/{dataset_id}/rolls/import',
    '/{dataset_id}/roll-numbers/import',
    '/{dataset_id}/assign-rolls',
    '/{dataset_id}/cloth-rolls',
):
    router.add_api_route(
        _path, upsert_rolls, methods=['POST'], status_code=201,
        name=f'register_rolls_{_path}',
    )


@router.put('/{dataset_id}/rolls/{lot_id}')
@router.post('/{dataset_id}/lots/{lot_id}/roll')
@router.put('/{dataset_id}/lots/{lot_id}/roll-number')
async def register_one_roll(dataset_id: int, lot_id: str, request: Request, session: Db, actor: Writer):
    content_type = request.headers.get('content-type', '')
    raw = await request.body()
    if not raw:
        payload = {}
    elif 'application/json' in content_type or raw.decode('utf-8-sig').lstrip().startswith(('{', '"')):
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {'roll_id': raw.decode().strip().strip('"')}
    else:
        payload = {'roll_id': raw.decode('utf-8-sig').strip()}
    if isinstance(payload, str):
        payload = {'roll_id': payload}
    if isinstance(payload, dict):
        payload.setdefault('lot_id', lot_id)
        payload.setdefault('roll_id', payload.get('roll_number') or payload.get('roll_no'))
    return register_rolls(session, dataset_id, payload, actor.id)


@router.patch('/{dataset_id}')
async def patch_dataset(dataset_id: int, request: Request, session: Db, actor: Writer):
    payload = await request.json()
    if not isinstance(payload, dict) or not any(
        key in payload for key in (
            'roll_numbers', 'roll_ids', 'cloth_roll_ids', 'lot_rolls',
            'roll_map', 'rolls', 'roll_links', 'links', 'assignments',
        )
    ):
        raise ValueError('当前仅支持通过该接口补登卷号。')
    return register_rolls(session, dataset_id, payload, actor.id)


@router.post('', response_model=DatasetInfo, status_code=201)
def insert(payload: DatasetImport, session: Db, actor: Writer):
    return import_dataset(session, payload, actor.id)
