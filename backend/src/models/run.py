from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class Run(Base):
    __tablename__ = 'analysis_runs'

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(140))
    dataset_id: Mapped[int] = mapped_column(ForeignKey('datasets.id'), index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
    created_at: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True)
    settings: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text)

    # Immutable grouping used when calculation started.  The natural key makes
    # identical data + grouping + parameters an idempotent evaluation.
    grouping: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    grouping_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), nullable=True, unique=True, index=True)
    invalidated_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def input_hash(self) -> str | None:
        return self.idempotency_key

    @property
    def fingerprint(self) -> str | None:
        return self.idempotency_key or self.grouping_key

    @property
    def is_invalidated(self) -> bool:
        return self.status == 'invalidated'

    @property
    def is_stale(self) -> bool:
        return self.status == 'invalidated'
