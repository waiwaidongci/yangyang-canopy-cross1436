from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class Dataset(Base):
    __tablename__ = 'datasets'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(140), unique=True)
    checksum: Mapped[str] = mapped_column(String(64))
    rows: Mapped[list] = mapped_column(JSON)
    # Cached canonical link list.  RollLink rows are the transactional source;
    # these JSON fields also keep lightweight/directly constructed datasets usable.
    roll_numbers: Mapped[list | dict | None] = mapped_column(JSON, nullable=True, default=list)
    roll_assignments: Mapped[list | dict | None] = mapped_column(JSON, nullable=True, default=list)
    roll_groups: Mapped[list | dict | None] = mapped_column(JSON, nullable=True, default=list)
    created_by: Mapped[int] = mapped_column(ForeignKey('accounts.id'))
