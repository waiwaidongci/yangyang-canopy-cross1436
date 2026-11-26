"""Evaluate the two predictions on exactly the same unseen batches."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .features import CrossModel, Score, pair_of
from .records import SeamRecord


@dataclass(frozen=True)
class ScoredSeam:
    row: SeamRecord
    score: Score


@dataclass(frozen=True)
class BatchErrors:
    lot_id: str
    rows: int
    crossed: int
    main_mae: float
    cross_mae: float

    @property
    def gain(self) -> float:
        return self.main_mae - self.cross_mae


@dataclass(frozen=True)
class HoldoutEvaluation:
    records: list[ScoredSeam]
    batches: list[BatchErrors]

    @property
    def main_mae(self) -> float:
        return mean(batch.main_mae for batch in self.batches)

    @property
    def cross_mae(self) -> float:
        return mean(batch.cross_mae for batch in self.batches)

    @property
    def gain(self) -> float:
        return self.main_mae - self.cross_mae

    @property
    def coverage(self) -> float:
        return 100 * sum(batch.crossed for batch in self.batches) / len(self.records)


def score_records(model: CrossModel, rows: list[SeamRecord]) -> list[ScoredSeam]:
    return [ScoredSeam(row, model.predict(pair_of(row))) for row in rows]


def evaluate_holdout(model: CrossModel, holdout: list[SeamRecord]) -> HoldoutEvaluation:
    scored = score_records(model, holdout)
    grouped: dict[str, list[ScoredSeam]] = defaultdict(list)
    for item in scored:
        grouped[item.row["lot_id"]].append(item)

    batches: list[BatchErrors] = []
    for lot, items in sorted(grouped.items()):
        main_errors: list[float] = []
        cross_errors: list[float] = []
        for item in items:
            actual = item.row["wrinkle_mm"]
            assert actual is not None
            main_errors.append(abs(actual - item.score.main_mm))
            cross_errors.append(abs(actual - item.score.cross_mm))
        batches.append(BatchErrors(
            lot, len(items), sum(item.score.uses_cross for item in items),
            mean(main_errors), mean(cross_errors),
        ))
    return HoldoutEvaluation(scored, batches)
