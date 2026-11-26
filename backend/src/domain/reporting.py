"""Review tables and a coefficient export for reproducing individual scores."""

from collections import Counter

from .evaluation import HoldoutEvaluation, ScoredSeam
from .features import CROSS_TERM, EDGE_TERM, SEAM_TERM, CrossModel, Support, feature_key, pair_of


RECHECK_MM = 8.0


def _table(key: str, title: str, labels: dict[str, str], rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def _pairs(model: CrossModel, holdout: HoldoutEvaluation, pending: list[ScoredSeam]) -> dict:
    held_counts = Counter(pair_of(item.row) for item in holdout.records)
    pending_counts = Counter(pair_of(item.row) for item in pending)
    pairs = sorted(model.support.keys() | held_counts.keys() | pending_counts.keys())
    rows = []
    for seam, edge in pairs:
        pair = seam, edge
        support = model.support.get(pair, Support())
        offset = model.offsets.get(pair)
        rows.append({
            "seam_type": seam, "edge_finish": edge, "train_rows": support.rows,
            "train_lots": support.lots, "holdout_rows": held_counts[pair],
            "pending_rows": pending_counts[pair], "status": model.status(pair),
            "feature_key": feature_key(CROSS_TERM, *pair) if offset is not None else "",
            "offset_mm": round(offset, 4) if offset is not None else None,
        })
    return _table("cross_support", "缝型与布边组合支持度", {
        "seam_type": "缝型", "edge_finish": "布边处理", "train_rows": "训练缝位数",
        "train_lots": "训练批次数", "holdout_rows": "留出缝位数", "pending_rows": "待抽缝位数",
        "status": "启用或回退原因", "feature_key": "启用的特征键", "offset_mm": "交叉修正（mm）",
    }, rows)


def _batches(evaluation: HoldoutEvaluation) -> dict:
    rows = [{
        "lot_id": batch.lot_id, "samples": batch.rows, "crossed": batch.crossed,
        "main_mae_mm": round(batch.main_mae, 4), "cross_mae_mm": round(batch.cross_mae, 4),
        "gain_mm": round(batch.gain, 4),
    } for batch in evaluation.batches]
    return _table("holdout_lots", "整批留出误差对账", {
        "lot_id": "留出批次", "samples": "实测缝位数", "crossed": "启用交叉数",
        "main_mae_mm": "主效应 MAE（mm）", "cross_mae_mm": "交叉 MAE（mm）",
        "gain_mm": "MAE 改善（mm）",
    }, rows)


SCORE_LABELS = {
    "id": "缝位编号", "lot_id": "裁缝批次", "seam_type": "缝型", "edge_finish": "布边处理",
    "main_mm": "主效应估计（mm）", "cross_mm": "交叉方案估计（mm）", "status": "启用或回退原因",
}


def _score_row(item: ScoredSeam) -> dict:
    row = item.row
    return {
        "id": row["id"], "lot_id": row["lot_id"], "seam_type": row["seam_type"],
        "edge_finish": row["edge_finish"], "main_mm": round(item.score.main_mm, 4),
        "cross_mm": round(item.score.cross_mm, 4), "status": item.score.reason,
    }


def _held_records(evaluation: HoldoutEvaluation) -> dict:
    return _table("holdout_seams", "留出缝位实测与估计", {
        **SCORE_LABELS, "actual_mm": "实测峰高（mm）",
    }, [{**_score_row(item), "actual_mm": item.row["wrinkle_mm"]} for item in evaluation.records])


def _pending_records(pending: list[ScoredSeam]) -> dict:
    ordered = sorted(pending, key=lambda item: (-item.score.cross_mm, item.row["id"]))
    rows = []
    for rank, item in enumerate(ordered, start=1):
        if not item.score.uses_cross:
            action = "补抽该组合"
        elif item.score.cross_mm >= RECHECK_MM:
            action = "优先复检"
        else:
            action = "常规复检"
        rows.append({"priority": rank, **_score_row(item), "action": action})
    return _table("pending_queue", "待抽检缝位清单", {
        "priority": "按估计峰高排序", **SCORE_LABELS, "action": "抽检动作",
    }, rows)


def _coefficients(model: CrossModel) -> dict:
    rows = [{"feature_key": "intercept", "term": "截距", "coefficient_mm": model.intercept}]
    for kind, label, terms in ((SEAM_TERM, "缝型主效应", model.seams),
                              (EDGE_TERM, "布边主效应", model.edges)):
        for level, coefficient in sorted(terms.items()):
            rows.append({"feature_key": feature_key(kind, level), "term": label,
                         "coefficient_mm": coefficient})
    for pair, coefficient in sorted(model.offsets.items()):
        rows.append({"feature_key": feature_key(CROSS_TERM, *pair), "term": "交叉残差修正",
                     "coefficient_mm": coefficient})
    return _table("feature_terms", "可复用的特征和系数", {
        "feature_key": "特征键", "term": "系数类型", "coefficient_mm": "系数（mm）",
    }, rows)


def build_report(
    model: CrossModel, holdout: HoldoutEvaluation, pending: list[ScoredSeam],
) -> dict:
    fallback = sum(batch.rows - batch.crossed for batch in holdout.batches)
    gain_mm = round(holdout.gain, 4)
    worsened = sum(round(batch.gain, 4) < 0 for batch in holdout.batches)
    warnings = []
    if not model.offsets:
        warnings.append("当前训练支持门槛下没有可启用的交叉组合，所有缝位采用主效应估计。")
    if fallback:
        warnings.append(f"留出样本中有 {fallback} 条采用回退，启用范围见组合支持度表。")
    if gain_mm <= 0:
        warnings.append("交叉方案在本次整批留出中未降低总体误差，复检排序建议沿用主效应并补抽。")
    elif worsened:
        warnings.append(f"总体误差下降，但有 {worsened} 个留出批次误差上升，需逐批复核。")
    if not pending:
        warnings.append("当前没有待抽检批次，复检清单为空。")

    return {
        "summary": (
            f"训练阶段启用 {len(model.offsets)} 个交叉组合。两种方案共用主效应，"
            f"在 {len(holdout.batches)} 个留出批次的 {len(holdout.records)} 条实测缝位上对照，"
            "每批先算平均绝对误差，再对批次等权平均。MAE 改善为正表示交叉方案误差下降。"
        ),
        "metrics": {
            "留出批次数": len(holdout.batches), "主效应 MAE（mm）": round(holdout.main_mae, 4),
            "交叉 MAE（mm）": round(holdout.cross_mae, 4), "MAE 改善（mm）": gain_mm,
            "留出交叉覆盖率（%）": round(holdout.coverage, 2),
        },
        "tables": [
            _pairs(model, holdout, pending), _batches(holdout), _held_records(holdout),
            _pending_records(pending), _coefficients(model),
        ],
        "warnings": warnings,
    }
