"""API alias for clients that name persisted runs as evaluations."""

from fastapi import APIRouter

from src.controllers.dependencies import Actor, Admin, Db, Writer
from src.controllers.report import report_csv, report_detail
from src.schemas.reviews import ReviewCreate, ReviewInfo
from src.schemas.runs import RunCreate, RunInfo
from src.services import reviews, runs

router = APIRouter(prefix='/evaluations')


@router.get('', response_model=list[RunInfo])
def list_evaluations(session: Db, actor: Actor):
    return runs.list_runs(session, actor)


@router.post('', response_model=RunInfo, status_code=201)
def create_evaluation(payload: RunCreate, session: Db, actor: Writer):
    return runs.create_run(session, payload, actor)


@router.get('/{run_id}')
def evaluation_detail(run_id: int, session: Db, actor: Actor):
    return report_detail(session, run_id, actor)


@router.post('/{run_id}/review', response_model=ReviewInfo, status_code=201)
def review_evaluation(run_id: int, payload: ReviewCreate, session: Db, actor: Admin):
    return reviews.submit(session, run_id, payload, actor)


@router.get('/{run_id}/export/{key}')
def export_evaluation(run_id: int, key: str, session: Db, actor: Actor):
    return report_csv(session, run_id, key, actor)
