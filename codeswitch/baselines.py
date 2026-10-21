"""Naive baselines, trained and evaluated separately for each language pair.

B0  prior:  always / never predict a switch; duration = most common class
B1  lookup: P(switch | current language, run length, trailing punctuation); duration = mode per language
B2  char n-gram TF-IDF + logistic regression on the last 6 characters
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from .lid import find_units, langs_of
from .metrics import score


def feats(text, pair=None):
    units = find_units(text)
    langs = langs_of(units, pair)
    run = 1                                                    # units in the current language so far
    while run < len(langs) and langs[-1 - run] == langs[-1]:
        run += 1
    trailing = int(len(text.rstrip()) > units[-1].end())       # punctuation / digits after the last unit
    return (langs[-1], min(run, 6), trailing)


def _gold(va):
    return [e["switch"] for e in va], [e["duration"] for e in va]


def _b0(tr, va):
    du_mode = Counter(e["duration"] for e in tr if e["duration"] > 0).most_common(1)[0][0]
    va_sw, va_du = _gold(va)
    n = len(va)
    return (score([1] * n, va_sw, [du_mode] * n, va_du),
            score([0] * n, va_sw, [du_mode] * n, va_du))


def _b1(tag, tr, va, a=2.0):
    prior = float(np.mean([e["switch"] for e in tr]))
    du_mode = Counter(e["duration"] for e in tr if e["duration"] > 0).most_common(1)[0][0]
    sw_tab = defaultdict(lambda: [0, 0])    # feats -> [no-switch count, switch count]
    du_tab = defaultdict(Counter)           # current language -> duration counts (switch=1 only)
    for e in tr:
        key = feats(e["text"], tag)
        sw_tab[key][e["switch"]] += 1
        if e["duration"] > 0:
            du_tab[key[0]][e["duration"]] += 1
    sw_pred, du_pred = [], []
    for e in va:
        key = feats(e["text"], tag)
        n0, n1 = sw_tab.get(key, (0, 0))
        p = (n1 + a * prior) / (n0 + n1 + a)              # smooth toward the global prior
        sw_pred.append(int(p > prior))                    # threshold at the base rate (like class weights in XLM-R)
        du_pred.append(du_tab[key[0]].most_common(1)[0][0] if du_tab[key[0]] else du_mode)
    va_sw, va_du = _gold(va)
    return score(sw_pred, va_sw, du_pred, va_du)


def _b2(tr, va, k=6):
    tail = lambda t: t[-k:]                               # current unit + the punctuation just typed
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    X_tr = vec.fit_transform([tail(e["text"]) for e in tr])
    X_va = vec.transform([tail(e["text"]) for e in va])
    clf_sw = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr, [e["switch"] for e in tr])
    has_dur = np.array([e["duration"] > 0 for e in tr])
    clf_du = LogisticRegression(max_iter=1000).fit(X_tr[has_dur], [e["duration"] for e in tr if e["duration"] > 0])
    va_sw, va_du = _gold(va)
    return score(clf_sw.predict(X_va), va_sw, clf_du.predict(X_va), va_du)


def run_baselines(train_raw, val_raw):
    """Return {baseline_name: {pair: score_dict}}."""
    res = {"B0 always-switch": {}, "B0 never-switch": {}, "B1 lookup": {}, "B2 char-ngram LR": {}}
    for tag in C.PAIR_TAGS:
        tr = [e for e in train_raw if e["pair"] == tag]
        va = [e for e in val_raw if e["pair"] == tag]
        res["B0 always-switch"][tag], res["B0 never-switch"][tag] = _b0(tr, va)
        res["B1 lookup"][tag] = _b1(tag, tr, va)
        res["B2 char-ngram LR"][tag] = _b2(tr, va)
    return res
