from enum import StrEnum


class Role(StrEnum):
    ADMIN = 'admin'
    ANALYST = 'analyst'
    VIEWER = 'viewer'


class RunStatus(StrEnum):
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    INVALIDATED = 'invalidated'
    # Descriptive alias retained for clients/tests using stale terminology.
    STALE = 'stale'


class Decision(StrEnum):
    APPROVED = 'approved'
    REJECTED = 'rejected'
