"""Application entry points for the stage canopy seam workflow."""

from .evaluation import evaluate_holdout, score_records
from .features import fit_cross_model
from .fixtures import DEMO_FILENAME, demo_rows
from .records import FIELDS, HOLDOUT, PENDING, TRAIN, normalize_rows
from .reporting import build_report


__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]

SPEC = {
    "columns": list(FIELDS),
    "fields": FIELDS,
    "settings": [
        {"key": "min_rows", "label": "组合最少训练缝位数", "default": 6, "min": 2, "max": 40, "step": 1},
        {"key": "min_lots", "label": "组合最少训练批次数", "default": 2, "min": 2, "max": 8, "step": 1},
    ],
    "modules": {"datasets": "缝位抽检台账", "runs": "交叉构造运行", "results": "组外误差与支持度", "reviews": "复检交接"},
    "limitation": "估计针对同一张挂和测量口径下的新裁缝批次。交叉系数用于预测，复检仍需实测。",
    "dataset_name": DEMO_FILENAME,
    "recommendation": "先复核整批留出的误差与交叉覆盖率，再交接待抽检清单，按回退提示补抽组合。",
}


def analyze(rows: list[dict], settings: dict) -> dict:
    validated = sorted(normalize_rows(rows), key=lambda row: row["id"])
    parameters = {}
    for field in SPEC["settings"]:
        value = settings.get(field["key"], field["default"])
        if (type(value) not in (int, float) or not field["min"] <= value <= field["max"]
                or int(value) != value):
            raise ValueError(f"「{field['label']}」必须为 {field['min']}～{field['max']} 的整数。")
        parameters[field["key"]] = int(value)

    train = [row for row in validated if row["split"] == TRAIN]
    holdout = [row for row in validated if row["split"] == HOLDOUT]
    pending = [row for row in validated if row["split"] == PENDING]
    model = fit_cross_model(train, **parameters)
    return build_report(model, evaluate_holdout(model, holdout), score_records(model, pending))
