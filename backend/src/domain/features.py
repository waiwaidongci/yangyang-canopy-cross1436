"""Frozen training vocabulary, sparse cross indicators and prediction fallback."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from .records import MAX_WRINKLE_MM, SeamRecord


Pair = tuple[str, str]
SEAM_TERM = "seam"
EDGE_TERM = "edge"
CROSS_TERM = "cross"
MAIN_PENALTY = 1.0
CROSS_PENALTY = 4.0
SUPPORTED = "启用交叉"
LOW_ROWS = "训练样本不足"
LOW_LOTS = "训练批次不足"
LOW_BOTH = "训练样本及批次不足"
UNSEEN_PAIR = "训练未见组合"
NEW_LEVEL = "训练未见单项，使用已知单项"
NEW_LEVELS = "训练未见两项，使用截距"


def pair_of(row: SeamRecord) -> Pair:
    return row["seam_type"], row["edge_finish"]


def feature_key(kind: str, *levels: str) -> str:
    return f"{kind}:{json.dumps(levels, ensure_ascii=True, separators=(',', ':'))}"


@dataclass(frozen=True)
class Support:
    rows: int = 0
    lots: int = 0

    def status(self, min_rows: int, min_lots: int) -> str:
        if self.rows == 0:
            return UNSEEN_PAIR
        if self.rows < min_rows and self.lots < min_lots:
            return LOW_BOTH
        if self.rows < min_rows:
            return LOW_ROWS
        if self.lots < min_lots:
            return LOW_LOTS
        return SUPPORTED


@dataclass(frozen=True)
class Score:
    main_mm: float
    cross_mm: float
    reason: str

    @property
    def uses_cross(self) -> bool:
        return self.reason == SUPPORTED


@dataclass(frozen=True)
class CrossModel:
    intercept: float
    seams: dict[str, float]
    edges: dict[str, float]
    support: dict[Pair, Support]
    offsets: dict[Pair, float]
    min_rows: int
    min_lots: int

    def raw_main(self, pair: Pair) -> float:
        seam, edge = pair
        return self.intercept + self.seams.get(seam, 0.0) + self.edges.get(edge, 0.0)

    def status(self, pair: Pair) -> str:
        seam, edge = pair
        if seam not in self.seams and edge not in self.edges:
            return NEW_LEVELS
        if seam not in self.seams or edge not in self.edges:
            return NEW_LEVEL
        return self.support.get(pair, Support()).status(self.min_rows, self.min_lots)

    def predict(self, pair: Pair) -> Score:
        main = self.raw_main(pair)
        cross = main + self.offsets.get(pair, 0.0)
        return Score(_clip(main), _clip(cross), self.status(pair))


def _clip(value: float) -> float:
    return min(MAX_WRINKLE_MM, max(0.0, value))


def fit_cross_model(train: list[SeamRecord], min_rows: int, min_lots: int) -> CrossModel:
    """Fit the common additive baseline, then shrink supported training residuals."""
    pairs = [pair_of(row) for row in train]
    target = [row["wrinkle_mm"] for row in train]
    design = [
        {feature_key(SEAM_TERM, seam): 1.0, feature_key(EDGE_TERM, edge): 1.0}
        for seam, edge in pairs
    ]
    encoder = DictVectorizer(sparse=True)
    matrix = encoder.fit_transform(design)
    baseline = Ridge(alpha=MAIN_PENALTY, solver="lsqr", tol=1e-10).fit(matrix, target)
    coefficients = dict(zip(encoder.get_feature_names_out(), map(float, baseline.coef_)))
    seams = {seam: coefficients[feature_key(SEAM_TERM, seam)] for seam, _ in sorted(set(pairs))}
    edges = {edge: coefficients[feature_key(EDGE_TERM, edge)] for _, edge in sorted(set(pairs))}

    row_counts = Counter(pairs)
    lots_by_pair: dict[Pair, set[str]] = defaultdict(set)
    for row, pair in zip(train, pairs):
        lots_by_pair[pair].add(row["lot_id"])
    support = {
        pair: Support(count, len(lots_by_pair[pair])) for pair, count in row_counts.items()
    }
    residuals: dict[Pair, list[float]] = {
        pair: [] for pair, counts in support.items() if counts.status(min_rows, min_lots) == SUPPORTED
    }
    for pair, actual, fitted in zip(pairs, target, baseline.predict(matrix)):
        if pair in residuals:
            residuals[pair].append(float(actual - fitted))
    offsets = {
        pair: fsum(values) / (len(values) + CROSS_PENALTY) for pair, values in residuals.items()
    }
    return CrossModel(float(baseline.intercept_), seams, edges, support, offsets, min_rows, min_lots)
