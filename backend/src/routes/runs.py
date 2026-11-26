from fastapi import APIRouter, Request

from src.controllers.dependencies import Actor, Admin, Db, Writer
from src.controllers.report import report_csv, report_detail
from src.schemas.reviews import ReviewCreate, ReviewInfo
from src.schemas.runs import RunCreate, RunInfo
from src.services import reviews, runs
from src.services.grouping import register_rolls

router = APIRouter(prefix='/runs')


@router.get('', response_model=list[RunInfo])
def list_all(session: Db, actor: Actor):
    return runs.list_runs(session, actor)


@router.post('', response_model=RunInfo, status_code=201)
def create(payload: RunCreate, session: Db, actor: Writer):
    return runs.create_run(session, payload, actor)


@router.get('/{run_id}')
def detail(run_id: int, session: Db, actor: Actor):
    return report_detail(session, run_id, actor)


@router.post('/{run_id}/rolls', status_code=201)
@router.patch('/{run_id}/roll-numbers')
@router.put('/{run_id}/roll-numbers')
async def register_run_rolls(run_id: int, request: Request, session: Db, actor: Writer):
    run, _ = runs.get_run(session, run_id, actor)
    try:
        payload = await request.json()
    except Exception:
        payload = (await request.body()).decode('utf-8-sig')
    return register_rolls(session, run.dataset_id, payload, actor.id)


@router.post('/{run_id}/invalidate')
def invalidate(run_id: int, session: Db, actor: Writer):
    return runs.invalidate_run(session, run_id)


@router.post('/{run_id}/review', response_model=ReviewInfo, status_code=201)
def review(run_id: int, payload: ReviewCreate, session: Db, actor: Admin):
    return reviews.submit(session, run_id, payload, actor)


@router.get('/{run_id}/export/{key}')
def export(run_id: int, key: str, session: Db, actor: Actor):
    return report_csv(session, run_id, key, actor)
