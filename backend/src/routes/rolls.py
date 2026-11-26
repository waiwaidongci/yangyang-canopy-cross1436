from fastapi import APIRouter, Request

from src.controllers.dependencies import Actor, Db, Writer
from src.services.grouping import grouping_overview, register_rolls

router = APIRouter()


def _dataset_id(payload, query_id=None):
    if query_id is not None:
        return query_id
    if isinstance(payload, dict):
        return payload.get('dataset_id') or payload.get('datasetId')
    return getattr(payload, 'dataset_id', None)


@router.get('/rolls')
@router.get('/roll-numbers')
@router.get('/roll-number')
def rolls_lookup(dataset_id: int, session: Db, actor: Actor):
    overview = grouping_overview(session, dataset_id)
    return {**overview, **(overview.get('roll_numbers') or {})}


async def _upsert(request: Request, session, actor, dataset_id: int | None = None):
    payload = await request.json()
    target_id = _dataset_id(payload, dataset_id)
    if not target_id:
        raise ValueError('补登卷号必须提供 dataset_id。')
    result = register_rolls(session, int(target_id), payload, actor.id)
    return {**result, **(result.get('roll_numbers') or {})}


@router.post('/rolls', status_code=201)
@router.post('/rolls/register', status_code=201)
@router.post('/roll-numbers', status_code=201)
@router.post('/roll-number', status_code=201)
@router.post('/roll-numbers/register', status_code=201)
async def rolls_upsert(request: Request, session: Db, actor: Writer, dataset_id: int | None = None):
    return await _upsert(request, session, actor, dataset_id)


@router.post('/lots/{lot_id}/roll-number', status_code=201)
@router.put('/lots/{lot_id}/roll-number')
@router.post('/lots/{lot_id}/roll', status_code=201)
async def lot_roll_upsert(lot_id: str, request: Request, session: Db, actor: Writer):
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {'roll_id': payload}
    dataset_id = payload.get('dataset_id') or payload.get('datasetId')
    if not dataset_id:
        raise ValueError('补登卷号必须提供 dataset_id。')
    payload['lot_id'] = lot_id
    return register_rolls(session, int(dataset_id), payload, actor.id)
