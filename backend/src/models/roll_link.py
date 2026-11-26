from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class RollLink(Base):
    """An idempotent retroactive roll-number registration for one lot."""

    __tablename__ = 'roll_links'
    __table_args__ = (
        UniqueConstraint('dataset_id', 'lot_id', 'roll_id', name='uq_roll_links_dataset_lot_roll'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey('datasets.id'), index=True)
    lot_id: Mapped[str] = mapped_column(String(80), index=True)
    roll_id: Mapped[str] = mapped_column(String(80), index=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey('accounts.id'), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))
