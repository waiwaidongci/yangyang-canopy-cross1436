from sqlalchemy import text

from src.config.database import Base, engine
from src.models import account, dataset, review, roll_link, run  # noqa: F401


def _existing_columns(table: str) -> set[str]:
    if engine.dialect.name != 'sqlite':
        return set()
    with engine.connect() as connection:
        rows = connection.execute(text(f'PRAGMA table_info({table})')).fetchall()
    return {row[1] for row in rows}


def _ensure_column(table: str, column: str, definition: str) -> None:
    columns = _existing_columns(table)
    if columns and column not in columns:
        with engine.begin() as connection:
            connection.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {definition}'))


def bootstrap():
    Base.metadata.create_all(engine)
    # create_all does not alter a persistent SQLite volume from the initial
    # snapshot.  These additive migrations keep that volume bootable.
    _ensure_column('datasets', 'roll_numbers', 'JSON')
    _ensure_column('datasets', 'roll_assignments', 'JSON')
    _ensure_column('datasets', 'roll_groups', 'JSON')
    for column, definition in (
        ('grouping', 'JSON'),
        ('grouping_key', 'VARCHAR(64)'),
        ('idempotency_key', 'VARCHAR(80)'),
        ('invalidated_at', 'VARCHAR(32)'),
        ('invalidation_reason', 'TEXT'),
    ):
        _ensure_column('analysis_runs', column, definition)
    if engine.dialect.name == 'sqlite':
        with engine.begin() as connection:
            connection.execute(text(
                'CREATE UNIQUE INDEX IF NOT EXISTS ux_analysis_runs_idempotency_key '
                'ON analysis_runs(idempotency_key)'
            ))
