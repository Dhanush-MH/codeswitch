"""Corpus loading, train/val split by conversation, and per-prefix example generation."""
import ast
import json
import random
import re

import numpy as np
import pandas as pd

from . import config as C
from .lid import find_units, langs_of, LANG_NAME, PIECE_RE


# ---------- Loading ----------
def parse_conv(s):
    """Parse the data_generation_result string into a list of sentences; None on failure."""
    if not isinstance(s, str):
        return None
    for loader in (ast.literal_eval, json.loads):
        try:
            x = loader(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(x, (list, tuple)):
            return [t for t in x if isinstance(t, str)]
    return None


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def load_records(path=C.DATA_PATH, verbose=True):
    """Return one record per sentence: {"pair", "conv", "text"}."""
    df = pd.read_csv(path)
    if verbose:
        print(df[["first_language", "second_language"]].value_counts().to_string())

    records, bad_rows = [], []
    for (l1, l2), tag in C.PAIRS.items():
        sub = df[(df["first_language"] == l1) & (df["second_language"] == l2)]
        if len(sub) == 0:
            raise ValueError(f"No {l1}-{l2} rows in the CSV; check PAIRS against the values printed above")
        if C.MAX_CONV_PER_PAIR is not None and len(sub) > C.MAX_CONV_PER_PAIR:
            sub = sub.sample(C.MAX_CONV_PER_PAIR, random_state=C.SEED)
        for row_id, s in zip(sub.index, sub["data_generation_result"]):
            sents = parse_conv(s)
            if sents is None:
                bad_rows.append((tag, row_id, s))
                continue
            for t in map(clean, sents):
                if len(find_units(t)) >= C.MIN_UNITS:
                    records.append({"pair": tag, "conv": f"{tag}#{row_id}", "text": t})

    with open(C.BAD_PATH, "w", encoding="utf-8") as f:
        for tag, row_id, s in bad_rows:
            f.write(f"=== pair={tag} | row_id={row_id} | type={type(s).__name__} ===\n{s}\n\n")
    if verbose:
        print(f"\n{len(records)} sentences; {len(bad_rows)} conversations failed to parse -> {C.BAD_PATH}")
    return records


def corpus_report(records, examples_per_pair=2):
    """Per-pair sanity check. If L1 units are almost absent, LID does not fit that pair
    (e.g. romanized Hindi, or the language column does not match the text)."""
    for tag in C.PAIR_TAGS:
        rs = [r for r in records if r["pair"] == tag]
        lg = [langs_of(find_units(r["text"]), tag) for r in rs]
        n_l1 = sum(l.count(0) for l in lg); n_en = sum(l.count(1) for l in lg)
        mixed = np.mean([len(set(l)) > 1 for l in lg])
        print(f"{tag:7} sentences {len(rs):6d} | L1 units {n_l1:8d} | EN units {n_en:8d} | sentences with a switch {mixed:.1%}")
        if n_l1 < C.L1_MIN_SHARE * (n_l1 + n_en):
            print(f"  !! {tag}: L1 share below threshold; check for romanized text or a wrong language column")
    for tag in C.PAIR_TAGS:
        for r in [r for r in records if r["pair"] == tag][:examples_per_pair]:
            print(tag, "|", r["text"])


# ---------- Split ----------
def split_records(records):
    """Split by conversation (VAL_FRAC per pair) so no conversation appears on both sides."""
    rng = random.Random(C.SEED)
    val_convs = set()
    for tag in C.PAIR_TAGS:
        cs = sorted({r["conv"] for r in records if r["pair"] == tag})
        rng.shuffle(cs)
        val_convs |= set(cs[:max(1, int(len(cs) * C.VAL_FRAC))])
    train = [r for r in records if r["conv"] not in val_convs]
    val = [r for r in records if r["conv"] in val_convs]
    return train, val


# ---------- Examples ----------
def make_examples(text, pair=None):
    """One example per stopping point (after each unit, and after each punctuation mark / number)."""
    units = find_units(text)
    n = len(units)
    langs = langs_of(units, pair)

    # run_len[j] = number of consecutive same-language units starting at unit j
    run_len = [1] * n
    for j in range(n - 2, -1, -1):
        if langs[j] == langs[j + 1]:
            run_len[j] = run_len[j + 1] + 1

    examples = []
    for i in range(n - 1):
        switch = int(langs[i] != langs[i + 1])
        # On a switch, unit i+1 starts the new segment, so run_len[i+1] is its length
        duration = C.dur_class(run_len[i + 1]) if switch else 0

        gap_start = units[i].end()
        gap = text[gap_start:units[i + 1].start()]                # punctuation / digits / spaces between units
        stops = [gap_start] + [gap_start + m.end() for m in PIECE_RE.finditer(gap)]
        for end in stops:
            examples.append({
                "text": text[:end],
                "pair": pair,
                "switch": switch,
                "duration": duration,
                "last_lang": LANG_NAME[langs[i]],                 # LID labels, for inspection only
                "next_lang": LANG_NAME[langs[i + 1]],
            })
    return examples


def build_examples(recs):
    return [e for r in recs for e in make_examples(r["text"], r["pair"])]


def example_report(train_raw, val_raw):
    sw = np.array([e["switch"] for e in train_raw])
    du = np.array([e["duration"] for e in train_raw])
    print(f"train: {len(train_raw)} examples, switch=1 share {sw.mean():.1%}")
    print("duration class counts:", np.bincount(du, minlength=C.N_DUR).tolist())
    print(f"val:   {len(val_raw)} examples")
    for tag in C.PAIR_TAGS:
        tr = [e for e in train_raw if e["pair"] == tag]; va = [e for e in val_raw if e["pair"] == tag]
        print(f"  {tag:7} train {len(tr):7d} (switch=1 {np.mean([e['switch'] for e in tr]):.1%}) | val {len(va):6d}")
