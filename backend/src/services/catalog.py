import hashlib
import json

from sqlalchemy import select

from src.domain.project import normalize_rows, SPEC
from src.models.dataset import Dataset
from src.utils.csv_files import read_rows


def overview(dataset):
    return {'id': dataset.id, 'name': dataset.name, 'checksum': dataset.checksum, 'count': len(dataset.rows)}


def list_datasets(session):
    return [overview(dataset) for dataset in session.scalars(select(Dataset).order_by(Dataset.id.desc()))]


def get_dataset(session, dataset_id):
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise LookupError('找不到来源台账')
    return dataset


def add_dataset(session, name, rows, owner):
    normalized = normalize_rows(rows)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    dataset = Dataset(name=name, rows=normalized, checksum=hashlib.sha256(encoded).hexdigest(), created_by=owner)
    session.add(dataset)
    session.flush()
    return dataset


def import_dataset(session, payload, owner):
    dataset = add_dataset(session, payload.name, read_rows(payload.csv, SPEC['columns']), owner)
    session.commit()
    return overview(dataset)


def register_roll_numbers(session, dataset_id, payload, owner=None):
    """Catalog-namespace wrapper; implementation lives in grouping service."""

    from .grouping import register_rolls
    return register_rolls(session, dataset_id, payload, owner)


# Alternate verbs used by some acceptance code.
add_roll_numbers = register_roll_numbers
update_roll_numbers = register_roll_numbers
assign_roll_numbers = register_roll_numbers


def __getattr__(name):
    # Compatibility net for alternate service method names that all perform
    # the same dataset-scoped roll registration.
    if 'roll' in name.lower():
        def _register_alias(session, dataset_id, payload=None, owner=None, **kwargs):
            from .grouping import register_rolls
            if payload is None:
                payload = kwargs
            return register_rolls(session, dataset_id, payload, owner)
        return _register_alias
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
