"""Unit segmentation and language identification (LID).

LID labels: 0 = non-English first language (L1), 1 = English.
Non-Latin scripts are identified by script; Latin-script pairs use wordfreq dictionaries.
"""
import re

from wordfreq import zipf_frequency

# ---------- Unit segmentation ----------
HAN    = r"[\u4e00-\u9fff\u3400-\u4dbf]"                      # Han: one character per unit (zh / yue / ja kanji)
KANA   = r"[\u3040-\u309f\u30a0-\u30ff]"                      # Japanese kana: one character per unit
DEV    = r"[\u0900-\u0963\u0971-\u097f][\u0900-\u0963\u0971-\u097f\u200c\u200d]*"   # Devanagari: one word
HANGUL = r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+"        # Korean: a run of Hangul is one word
CYR    = r"[\u0400-\u04ff]+(?:-[\u0400-\u04ff]+)*"            # Cyrillic: one word
ARA    = r"[\u0621-\u064a\u066e-\u06d3\u06fa-\u06ff][\u0621-\u064a\u0640\u064b-\u065f\u0670\u066e-\u06d3\u06fa-\u06ff]*"  # Arabic: one word
LATC   = "A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff"          # Latin letters incl. Vietnamese diacritics
LAT    = rf"[{LATC}]+(?:['’][{LATC}]+)*"

UNIT_RE = re.compile(f"{HAN}|{KANA}|{DEV}|{HANGUL}|{CYR}|{ARA}|{LAT}")
LAT_RE = re.compile(LAT)
PIECE_RE = re.compile(r"\d+|[^\s\d]")      # a number or a single punctuation mark
NON_LATIN_PAIRS = {"zh-en", "ja-en", "ko-en", "hi-en", "ru-en", "ar-en"}
LANG_NAME = {0: "L1", 1: "en"}             # in de-fr, label 1 is actually French


def find_units(text):
    return list(UNIT_RE.finditer(text))


def lang_of(unit):
    """Non-Latin script -> 0 (L1); Latin script -> 1 (English)."""
    return 1 if LAT_RE.match(unit.group()) else 0


# ---------- Latin-script pairs: dictionary LID ----------
# pair -> (L1 dictionary, L2 dictionary, chars only in L1, chars only in L2)
_c = re.compile
LATIN_PAIRS = {
    "es-en":  ("es",  "en", _c(r"[áéíóúñü¿¡]"),         None),
    "fr-en":  ("fr",  "en", _c(r"[àâçéèêëîïôûùüÿœæ]"),  None),
    "it-en":  ("it",  "en", _c(r"[àèéìíîòóùú]"),        None),
    "de-en":  ("de",  "en", _c(r"[äöüß]"),              None),
    "de-fr":  ("de",  "fr", _c(r"[äöß]"),               _c(r"[àâçéèêëîïôûùÿœæ]")),
    "vi-en":  ("vi",  "en", _c(r"[^\x00-\x7f]"),        None),   # Vietnamese: any word with a diacritic
    #"hi-en":  ("hi",  "en", None,                       None),   # To-do: handles romanized Hindi
    "ms-en":  ("ms",  "en", None,                       None),
    "fil-en": ("fil", "en", None,                       None),
}
_lid_cache = {}


def _word_lid(w, pair):
    """0 = L1, 1 = L2, None = ambiguous."""
    key = (pair, w.lower())
    if key not in _lid_cache:
        c1, c2, o1, o2 = LATIN_PAIRS[pair]
        wl = w.lower()
        if o1 and o1.search(wl):
            r = 0
        elif o2 and o2.search(wl):
            r = 1
        else:
            d = zipf_frequency(wl, c1) - zipf_frequency(wl, c2)
            r = 0 if d >= 1 else (1 if d <= -1 else None)
        _lid_cache[key] = r
    return _lid_cache[key]


def _word_d(w, pair):
    """Frequency difference, used as a tiebreak when left and right neighbours disagree."""
    c1, c2, _, _ = LATIN_PAIRS[pair]
    return zipf_frequency(w.lower(), c1) - zipf_frequency(w.lower(), c2)


def langs_of(units, pair=None):
    """Per-unit language labels. Ambiguous words copy the nearest determined neighbour."""
    if pair not in LATIN_PAIRS:
        return [lang_of(u) for u in units]
    raw = [_word_lid(u.group(), pair) for u in units]
    out, n, i = list(raw), len(raw), 0
    while i < n:
        if raw[i] is not None:
            i += 1
            continue
        j = i
        while j < n and raw[j] is None:
            j += 1
        L = raw[i - 1] if i > 0 else None
        R = raw[j] if j < n else None
        for k in range(i, j):
            if L is None and R is None: out[k] = 1
            elif L is None:             out[k] = R
            elif R is None:             out[k] = L
            elif L == R:                out[k] = L
            else:                       out[k] = 0 if _word_d(units[k].group(), pair) > 0 else 1
        i = j
    return out


SELFTEST = [
    ("zh-en", "我喜欢 this movie，因为 it's 很好看。"),
    ("yue-en", "佢今日 meeting 完咗先返屋企。"),
    ("hi-en", "मुझे यह movie बहुत पसंद आई, because it was amazing!"),
    ("hi-en", "mujhe yeh movie bahut pasand aayi because it was amazing!"),   # romanized
    ("es-en", "Ayer fui al cine con mi amigo, and we watched a really good movie, pero no me gustó el final."),
    ("ar-en", "أنا أحب هذا movie كثيراً because it's great"),
    ("ja-en", "今日はmeetingがあるのでI'll be late"),
    ("ko-en", "나는 이 movie가 정말 좋아 because it was fun"),
    ("ru-en", "Я вчера watched a movie и мне понравилось"),
    ("vi-en", "Hôm nay tôi đi to the office rồi về nhà"),
    ("fr-en", "Je suis allé au cinema with my friend mais c'était nul"),
    ("de-fr", "Ich habe gestern un film très intéressant gesehen"),
    ("fil-en", "Pumunta ako sa mall kahapon and I bought new shoes"),
]


def selftest():
    """Print the (unit, label) sequence for one example sentence per pair."""
    for tag, t in SELFTEST:
        us = find_units(t)
        print(tag, [(u.group(), l) for u, l in zip(us, langs_of(us, tag))])
