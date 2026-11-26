from pydantic import BaseModel, ConfigDict, Field
from typing import Any

from src.constants.workflow import RunStatus


class RunCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    label: str = Field(min_length=2, max_length=140)
    dataset_id: int = Field(gt=0)
    settings: dict[str, float] = Field(default_factory=dict, max_length=20)

    # Optional explicit grouping inputs.  The normal API workflow stores roll
    # links on the dataset and omits these from /runs.
    grouping: Any | None = None
    roll_numbers: Any | None = None
    roll_ids: Any | None = None
    cloth_roll_ids: Any | None = None
    lot_rolls: Any | None = None
    roll_map: Any | None = None
    roll_links: list[dict[str, Any]] | None = None
    rolls: Any | None = None
    groups: Any | None = None
    roll_groups: Any | None = None
    selection_mode: str | None = None
    mode: str | None = None
    strategy: str | None = None
    global_selection: bool | None = None
    preserve_unmixed: bool | None = None
    idempotency_key: str | None = Field(default=None, max_length=120)


class RunInfo(BaseModel):
    id: int
    label: str
    dataset_id: int
    created_at: str
    status: RunStatus
    settings: dict
    decision: str | None
    grouping: dict | None = None
    grouping_key: str | None = None
    idempotency_key: str | None = None
    invalidated_at: str | None = None
    invalidation_reason: str | None = None
    replayed: bool | None = None
    valid: bool | None = None
    invalidated: bool | None = None
    stale: bool | None = None
