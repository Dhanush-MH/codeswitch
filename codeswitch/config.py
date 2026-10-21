"""Central configuration. Edit here; the notebook uses %autoreload so changes apply on re-run."""

IGNORE = -100          # label value ignored by the losses
SEED = 42

# ---- Data ----
DATA_PATH = "SwitchLingua_text.csv"
BAD_PATH = "parse_failed_conversations.txt"

# (first_language, second_language) -> pair tag. Names must match the CSV values exactly.
PAIRS = {
    ("Hindi", "English"):      "hi-en",
    ("Cantonese", "English"):  "yue-en",
    ("Spanish", "English"):    "es-en",
    # ("Arabic", "English"):     "ar-en",
    # ("German", "English"):     "de-en",
    # ("Vietnamese", "English"): "vi-en",
    # ("Japanese", "English"):   "ja-en",
    # ("Korean", "English"):     "ko-en",
    # ("French", "English"):     "fr-en",
    # ("Russian", "English"):    "ru-en",
    # ("Italian", "English"):    "it-en",
    # ("Chinese", "English"):    "zh-en",
}

PAIR_TAGS = list(PAIRS.values())
PAIR2ID = {t: i for i, t in enumerate(PAIR_TAGS)}
MAX_CONV_PER_PAIR = 1000   # max conversations per pair; None = use all
L1_MIN_SHARE = 0.1        # corpus_report warns when the L1 unit share falls below this
MIN_UNITS = 2             # a sentence needs at least this many units to have a prediction point
VAL_FRAC = 0.1            # fraction of conversations (per pair) held out for validation

# ---- Labels ----
N_DUR = 4                 # 0 = no switch; 1: 1-2 units; 2: 3-7; 3: >=8
def dur_class(run):
    """Duration class for a new segment of `run` units (1..3)."""
    return 1 + (run > 2) + (run > 7)

LAMBDA_DUR = 3          # weight of the duration loss

# ---- XLM-R ----
XLMR_NAME = "xlm-roberta-base"
MAX_LEN = 128             # safety truncation (left side is dropped)
EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-5
HEAD_LR = 1e-3
KEEP_NEG = 0.3            # fraction of switch=0 training examples kept (switch=1 always kept); 1.0 = no downsampling
BUCKET_MULT = 50          # length bucketing: sort within chunks of BATCH_SIZE * BUCKET_MULT examples
VAL_BATCH_SIZE = 64

# ---- Qwen3 ----
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
QWEN_MAX_LEN = 256        # Devanagari takes many Qwen tokens, so this is looser than MAX_LEN
QWEN_BATCH_SIZE = 16
QWEN_EPOCHS = 4
QWEN_LR = 1e-5
QWEN_HEAD_LR = 1e-4
DUR_SMOOTH = 0.1          # label smoothing for the duration loss
LAMBDA_ORD = 0.5          # ordinal L1 penalty on the expected duration class
