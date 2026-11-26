"""Reproducible panels for the autumn stage canopy inspection ledger."""

from random import Random

from .records import HOLDOUT, PENDING, TRAIN, SeamRecord, normalize_rows


DEMO_FILENAME = "绒岸-秋演天幕缝位抽检.csv"
COMMON_EFFECTS = {
    ("平缝", "折边"): -2.5,
    ("平缝", "锁边"): 3.2,
    ("平缝", "热切边"): -0.7,
    ("搭接缝", "折边"): 3.4,
    ("搭接缝", "锁边"): -2.8,
    ("搭接缝", "热切边"): -0.6,
    ("暗扣缝", "折边"): -0.9,
    ("暗扣缝", "锁边"): 1.1,
    ("包带缝", "锁边"): -0.4,
}
SEAM_EFFECTS = {"平缝": 0.0, "搭接缝": 0.7, "暗扣缝": -0.5, "包带缝": 0.6, "压胶缝": 1.2}
EDGE_EFFECTS = {"折边": 0.2, "锁边": 0.0, "热切边": 0.5, "涂胶边": 0.8}
BATCH_ONLY_PAIR = ("包带缝", "热切边")
LOW_COUNT_PAIR = ("包带缝", "折边")
NEW_PAIR = ("暗扣缝", "热切边")
ONE_NEW_LEVEL = ("压胶缝", "折边")
TWO_NEW_LEVELS = ("压胶缝", "涂胶边")
EXTRA_EFFECTS = {BATCH_ONLY_PAIR: 1.1, LOW_COUNT_PAIR: -0.4, NEW_PAIR: 0.5}


def _measured_lot(
    random: Random, lot: str, split: str, pairs: list[tuple[str, str]], batch_index: int,
) -> list[SeamRecord]:
    lot_shift = 0.28 * ((batch_index % 5) - 2)
    records: list[SeamRecord] = []
    for index, (seam, edge) in enumerate(pairs, start=1):
        pair_effect = COMMON_EFFECTS.get((seam, edge), EXTRA_EFFECTS.get((seam, edge), 0.0))
        height = 6.4 + SEAM_EFFECTS[seam] + EDGE_EFFECTS[edge] + pair_effect + lot_shift
        records.append({
            "id": f"{lot}-{index:02d}", "lot_id": lot, "split": split,
            "seam_type": seam, "edge_finish": edge,
            "wrinkle_mm": round(height + random.uniform(-0.63, 0.63), 2),
        })
    return records


def demo_rows() -> list[SeamRecord]:
    random = Random(261125)
    common = list(COMMON_EFFECTS)
    rows: list[SeamRecord] = []
    for index in range(1, 11):
        pairs = common + [common[(index + 2 * step) % len(common)] for step in range(3)]
        if index in {1, 3}:
            pairs.append(LOW_COUNT_PAIR)
        if index == 2:
            pairs.extend([BATCH_ONLY_PAIR] * 8)
        rows.extend(_measured_lot(random, f"RA-T{index:02d}", TRAIN, pairs, index))

    extras = [LOW_COUNT_PAIR, BATCH_ONLY_PAIR, ONE_NEW_LEVEL, TWO_NEW_LEVELS]
    for index, extra in enumerate(extras, start=1):
        sample = common + [common[index], common[index + 3], NEW_PAIR, extra]
        if index == 4:
            sample.append(NEW_PAIR)
        rows.extend(_measured_lot(random, f"RA-H{index:02d}", HOLDOUT, sample, index + 10))

    pending_pairs = [*common[:5], BATCH_ONLY_PAIR, NEW_PAIR, ONE_NEW_LEVEL, TWO_NEW_LEVELS]
    for index in range(1, 4):
        lot = f"RA-P{index:02d}"
        for position, (seam, edge) in enumerate(pending_pairs, start=1):
            rows.append({
                "id": f"{lot}-{position:02d}", "lot_id": lot, "split": PENDING,
                "seam_type": seam, "edge_finish": edge, "wrinkle_mm": None,
            })
    return normalize_rows(rows)
