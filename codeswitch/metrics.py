"""Shared evaluation: switch P/R/F1, duration accuracy/MAE, per-pair breakdown, result tables."""
import json

import numpy as np
import pandas as pd

from . import config as C


def score(sw_pred, sw_gold, du_pred, du_gold):
    sw_pred, sw_gold, du_pred, du_gold = map(np.array, (sw_pred, sw_gold, du_pred, du_gold))
    # switch: positive class = language switch
    tp = int(((sw_pred == 1) & (sw_gold == 1)).sum())
    fp = int(((sw_pred == 1) & (sw_gold == 0)).sum())
    fn = int(((sw_pred == 0) & (sw_gold == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    # duration: evaluated only where the gold label is a switch
    m = du_gold > 0
    dur_acc = float((du_pred[m] == du_gold[m]).mean()) if m.any() else float("nan")
    dur_mae = float(np.abs(du_pred[m] - du_gold[m]).mean()) if m.any() else float("nan")
    return {"switch_P": round(p, 3), "switch_R": round(r, 3), "switch_F1": round(f1, 3),
            "dur_acc": round(dur_acc, 3), "dur_MAE": round(dur_mae, 3)}


def score_by_pair(sw_pred, sw_gold, du_pred, du_gold, pairs):
    """Return {"ALL": ..., "<pair>": ...}."""
    sw_pred, sw_gold, du_pred, du_gold, pairs = map(np.array, (sw_pred, sw_gold, du_pred, du_gold, pairs))
    out = {"ALL": score(sw_pred, sw_gold, du_pred, du_gold)}
    for tag in C.PAIR_TAGS:
        m = pairs == tag
        if m.any():
            out[tag] = score(sw_pred[m], sw_gold[m], du_pred[m], du_gold[m])
    return out


def print_scores(prefix, res):
    for k, v in res.items():
        print(f"{prefix} [{k:6}]", v)


def results_table(results, metric="switch_F1"):
    """results: {model_name: {pair_or_ALL: score_dict}} -> DataFrame (rows = pairs, cols = models)."""
    df = pd.DataFrame({name: {k: v[metric] for k, v in res.items()} for name, res in results.items()})
    return df.reindex(["ALL"] + C.PAIR_TAGS)


def save_results(results, path="results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
