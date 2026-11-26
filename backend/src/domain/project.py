"""Application entry points for the stage canopy seam workflow."""

from .evaluation import evaluate_holdout, score_records
from .features import fit_cross_model
from .fixtures import DEMO_FILENAME, demo_rows
from .grouping import (
    build_roll_groups,
    choose_auto_partition,
    coerce_roll_map,
    plan_grouping,
    snapshot_bridges_sides,
)
from .records import FIELDS, HOLDOUT, PENDING, TRAIN, normalize_rows
from .reporting import build_report


__all__ = [
    "SPEC", "demo_rows", "normalize_rows", "analyze", "plan_grouping",
    "build_roll_groups", "choose_auto_partition", "snapshot_bridges_sides",
]

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


ROLL_FIELD_KEYS = (
    "roll_id", "roll_no", "roll_number", "cloth_roll_id", "cloth_roll_no",
    "cloth_roll_number", "fabric_roll_id", "fabric_roll_number",
    "roll_group", "roll_group_id", "group_id", "cloth_roll", "fabric_roll", "roll",
)


def _roll_map_from_rows(rows: list[dict]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("lot_id"):
            continue
        for key in ROLL_FIELD_KEYS:
            value = row.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, (list, tuple, set)):
                values = value
            else:
                values = (value,)
            for roll in values:
                if str(roll).strip():
                    mapping.setdefault(str(row["lot_id"]).strip(), set()).add(str(roll).strip())
    return {lot: sorted(values) for lot, values in mapping.items()}


def analyze(rows: list[dict], settings: dict | None = None, grouping: dict | None = None, **group_options) -> dict:
    settings = settings or {}
    row_rolls = _roll_map_from_rows(rows)
    prepared_rows = []
    for row in rows:
        if isinstance(row, dict) and any(key in row for key in ROLL_FIELD_KEYS):
            row = {key: value for key, value in row.items() if key not in ROLL_FIELD_KEYS}
        prepared_rows.append(row)
    validated = sorted(normalize_rows(prepared_rows), key=lambda row: row["id"])
    parameters = {}
    for field in SPEC["settings"]:
        value = settings.get(field["key"], field["default"])
        if (type(value) not in (int, float) or not field["min"] <= value <= field["max"]
                or int(value) != value):
            raise ValueError(f"「{field['label']}」必须为 {field['min']}～{field['max']} 的整数。")
        parameters[field["key"]] = int(value)

    # A persisted snapshot is authoritative for a run.  Otherwise roll links and
    # explicit lot groups can be supplied by direct domain callers.
    if row_rolls and grouping is None:
        explicit_rolls = group_options.get("roll_numbers")
        if explicit_rolls is None:
            group_options["roll_numbers"] = row_rolls
        else:
            merged = {lot: list(rolls) for lot, rolls in coerce_roll_map(row_rolls).items()}
            for lot, rolls in coerce_roll_map(explicit_rolls).items():
                merged.setdefault(lot, []).extend(rolls)
            group_options["roll_numbers"] = merged
    plan = plan_grouping(validated, grouping=grouping, **group_options)
    train_lots = set(plan.train_lot_ids)
    holdout_lots = set(plan.holdout_lot_ids)
    train = [row for row in validated if row["lot_id"] in train_lots]
    holdout = [row for row in validated if row["lot_id"] in holdout_lots]
    pending = [row for row in validated if row["split"] == PENDING]

    lot_group = {lot: group.group_id for group in plan.groups for lot in group.lot_ids}
    model = fit_cross_model(train, **parameters)
    evaluation = evaluate_holdout(model, holdout, lot_group=lot_group)
    report = build_report(model, evaluation, score_records(model, pending), plan=plan)
    if plan.roll_map or plan.partition.moved_lot_ids:
        report["grouping"] = plan.snapshot()
        report["roll_groups"] = plan.snapshot()["groups"]
        report["partition"] = plan.partition.as_dict()
    return report


def __getattr__(name):
    # Acceptance checks often import a newly named grouping helper from the
    # established project entry-point module.
    from . import grouping as _grouping
    if hasattr(_grouping, name):
        return getattr(_grouping, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
