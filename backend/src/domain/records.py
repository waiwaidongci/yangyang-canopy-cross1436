"""CSV boundary, physical measurement units and batch ownership."""

import math
from collections import defaultdict
from typing import TypedDict


FIELDS = {
    "id": "缝位编号",
    "lot_id": "裁缝批次",
    "split": "用途（train / holdout / pending）",
    "seam_type": "缝型",
    "edge_finish": "布边处理",
    "wrinkle_mm": "张挂后皱褶峰高（mm）",
}
TRAIN = "train"
HOLDOUT = "holdout"
PENDING = "pending"
MAX_WRINKLE_MM = 30.0


class SeamRecord(TypedDict):
    id: str
    lot_id: str
    split: str
    seam_type: str
    edge_finish: str
    wrinkle_mm: float | None


def _text(raw: object, key: str, line: int) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"第 {line} 条的「{FIELDS[key]}」必须填写文本。")
    value = raw.strip()
    if len(value) > 64 or any(not char.isprintable() for char in value):
        raise ValueError(f"第 {line} 条的「{FIELDS[key]}」应为 1～64 个可显示字符。")
    return value


def _measurement(raw: object, split: str, line: int) -> float | None:
    if split == PENDING:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        raise ValueError(f"第 {line} 条是待抽检记录，皱褶峰高必须留空。")
    if not isinstance(raw, (int, float, str)) or isinstance(raw, bool):
        raise ValueError(f"第 {line} 条必须填写数值型皱褶峰高。")
    try:
        value = float(raw)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"第 {line} 条必须填写数值型皱褶峰高。") from exc
    if not math.isfinite(value) or not 0 <= value <= MAX_WRINKLE_MM:
        raise ValueError(f"第 {line} 条的皱褶峰高必须为 0～{MAX_WRINKLE_MM:g} mm 的有限值。")
    return value


def normalize_rows(rows: list[dict]) -> list[SeamRecord]:
    """Validate the exact source schema and prevent one lot crossing partitions."""
    normalized: list[SeamRecord] = []
    seen_ids: set[str] = set()
    lot_splits: dict[str, str] = {}
    lot_sets: dict[str, set[str]] = defaultdict(set)
    row_counts: dict[str, int] = defaultdict(int)
    for line, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.keys() != FIELDS.keys():
            raise ValueError(f"第 {line} 条字段与导入格式不一致，应包含 {', '.join(FIELDS)}。")
        identifier = _text(row["id"], "id", line)
        lot = _text(row["lot_id"], "lot_id", line)
        split = _text(row["split"], "split", line)
        if identifier in seen_ids:
            raise ValueError(f"缝位编号「{identifier}」重复。")
        if split not in {TRAIN, HOLDOUT, PENDING}:
            raise ValueError(f"第 {line} 条用途必须是 train、holdout 或 pending。")
        if lot_splits.setdefault(lot, split) != split:
            raise ValueError(f"裁缝批次「{lot}」跨用途，同一批次的全部缝位必须归入同一用途。")
        normalized.append({
            "id": identifier,
            "lot_id": lot,
            "split": split,
            "seam_type": _text(row["seam_type"], "seam_type", line),
            "edge_finish": _text(row["edge_finish"], "edge_finish", line),
            "wrinkle_mm": _measurement(row["wrinkle_mm"], split, line),
        })
        seen_ids.add(identifier)
        lot_sets[split].add(lot)
        row_counts[split] += 1

    for split, label, min_rows in ((TRAIN, "训练", 18), (HOLDOUT, "留出", 9)):
        if len(lot_sets[split]) < 3 or row_counts[split] < min_rows:
            raise ValueError(f"{label}数据至少需要 3 个独立批次和 {min_rows} 条实测记录。")
    return normalized
