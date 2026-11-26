from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RollLinkCreate(BaseModel):
    model_config = ConfigDict(extra='allow', str_strip_whitespace=True)

    dataset_id: int | None = None
    lot_id: str | None = None
    roll_id: str | None = None
    roll_number: str | None = None
    roll_no: str | None = None
    lot: str | None = None
    batch_id: str | None = None
    roll: Any | None = None


class RollRegistrationCreate(BaseModel):
    model_config = ConfigDict(extra='allow', str_strip_whitespace=True)

    dataset_id: int | None = None
    roll_numbers: Any | None = None
    roll_ids: Any | None = None
    roll_map: Any | None = None
    rolls: Any | None = None
    roll_links: list[dict[str, Any]] | None = None
    links: list[dict[str, Any]] | None = None
    assignments: list[dict[str, Any]] | None = None


class RollNumberCreate(RollRegistrationCreate):
    """Compatibility name for retroactive roll-number entry."""


class RollNumberUpdate(RollRegistrationCreate):
    pass


class RollAssignmentCreate(RollRegistrationCreate):
    pass


class RollGroupCreate(BaseModel):
    model_config = ConfigDict(extra='allow')

    group_id: str | None = None
    lot_ids: list[str] = Field(default_factory=list)
    roll_ids: list[str] = Field(default_factory=list)


# Common alternate class names.
RollCreate = RollLinkCreate
RollLinkIn = RollLinkCreate
RollNumbersRequest = RollRegistrationCreate
RollRegistrationRequest = RollRegistrationCreate
