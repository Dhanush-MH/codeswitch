# codeswitch: Predictive Multitask Learning for Streaming Code-Switching

CS6140 Group Project (Team 6), Northeastern University, Khoury College of Computer Sciences, Seattle.

Given only the text typed so far, **will the next unit switch language, and how long will the new-language run last?** This repository trains and compares an encoder-only model (XLM-R) and a decoder-only model (Qwen3) on the SwitchLingua code-switching corpus, with simple baselines for reference.

Status: Project Update 1 (three language pairs: Hindi-English, Cantonese-English, Spanish-English).

## Task

At every stopping point of a sentence (after a unit, or after a punctuation mark or number) the model sees only the prefix and predicts two things with a shared encoder and two heads:

| Head | Target | Meaning |
|---|---|---|
| Switch | binary | does the **next** unit have a different language than the current one? |
| Duration | 4 classes | `0` = no switch; `1` = new-language run of 1-2 units; `2` = 3-7 units; `3` = 8 or more units |

Language IDs are used only to **generate** the labels. They are never model inputs. At evaluation the duration is read only from classes 1-3, and duration accuracy is computed only at positions where a switch really occurs.

Units: Han characters and kana count as one unit each; all other scripts use whitespace-separated words. Punctuation and numbers stay in the context and inherit the labels of the preceding unit.

## Data

SwitchLingua text corpus (synthetic, LLM-generated code-switched conversations; no intra-word switches):
https://huggingface.co/datasets/Shelton1013/SwitchLingua_text

Download it and save it as `SwitchLingua_text.csv` in the project root (or change `DATA_PATH` in `codeswitch/config.py`). Conversations that fail to parse are listed in `parse_failed_conversations.txt`.

The active pairs are set in `codeswitch/config.py` (`PAIRS`). Uncomment a line there to enable another pair.

## Repository layout

```
pipeline.ipynb        main notebook: pipeline, results, commentary
codeswitch/
  config.py           hyperparameters, language pairs, paths
  utils.py            device, seeding, memory monitoring, class weights
  lid.py              unit segmentation + language identification
  data.py             corpus loading, split by conversation, example generation
  metrics.py          score / per-pair score / result tables
  baselines.py        B0 prior, B1 lookup table, B2 char n-gram + logistic regression
  trainer.py          shared training loop (differential learning rates, best-epoch restore)
  xlmr.py             XLM-R dataset, model, evaluate, predict
  qwen.py             Qwen3 dataset, model, evaluate, predict
requirements.txt
LICENSE               MIT
```

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The notebook picks the available device automatically (it was run on Apple Silicon, `mps`).

## How to run

1. Put `SwitchLingua_text.csv` in the project root.
2. Open `pipeline.ipynb` and run the cells in order:
   0. setup and seed
   1. segmentation and language-ID self-test
   2. load the corpus and print the corpus report
   3. split by conversation and build prefix examples
   4. baselines (B0, B1, B2)
   5. XLM-R training and evaluation
   6. Qwen3 training and evaluation
   7. comparison table; results are saved to `results.json`
3. Change settings in `codeswitch/config.py`; the notebook uses `%autoreload`, so edits apply when you re-run.

## Method in brief

- **Language ID:** Cantonese and Hindi use a script rule (Han/Devanagari = L1, Latin = English). Spanish uses word-frequency scoring (`wordfreq`) with ambiguous words resolved from the nearest determined neighbours. Latinized Hindi is not detected yet (to do).
- **Split:** 10% of each pair's conversations are held out for validation. Splitting is by conversation, so sentences of one conversation never appear in both sets.
- **XLM-R:** input is the prefix followed by a `<mask>` position; the two heads read that position. Negative examples are down-sampled (keep fraction `KEEP_NEG = 0.3`), with class weights computed on the down-sampled set and length-bucketed batches. Validation is never down-sampled.
- **Qwen3:** one sentence per sample. The causal mask gives predictions for every stopping point in one forward pass. The sentence is split at stopping points and each chunk tokenized separately, with labels on the last token of each chunk. Embeddings are frozen. The duration loss adds label smoothing and an ordinal L1 term.
- **Loss:** switch cross-entropy + `LAMBDA_DUR` (3) x duration cross-entropy + ordinal L1 penalty (`LAMBDA_ORD` = 0.5) on switch positions.
- **Baselines** (each trained per pair on the same split): B0 always predicts a switch; B1 is a lookup table on current language, run length and trailing punctuation; B2 is TF-IDF on the last 6 characters with logistic regression.

## Key settings (`codeswitch/config.py`)

| Setting | XLM-R | Qwen3 |
|---|---|---|
| Model | `xlm-roberta-base` | `Qwen/Qwen3-0.6B-Base` |
| Epochs | 5 | 4 |
| Batch size | 32 | 16 |
| Learning rate (encoder / heads) | 3e-5 / 1e-3 | 1e-5 / 1e-4 |
| Max length | 128 | 256 |

Shared: seed 42, up to 1,000 conversations per pair, validation fraction 0.1.

## Results (validation set, best epoch)

Data: 198,006 training and 21,844 validation examples (9.5% positive) over the three pairs.

**Switch F1**

| Model | hi-en | yue-en | es-en | pooled |
|---|---|---|---|---|
| B0 always-switch | 0.138 | 0.172 | 0.213 | - |
| B1 lookup | 0.310 | 0.417 | 0.361 | - |
| B2 char n-gram LR | 0.388 | 0.350 | 0.469 | - |
| XLM-R | 0.673 | 0.569 | 0.651 | 0.617 |
| Qwen3-0.6B | 0.697 | 0.606 | 0.700 | 0.658 |

**Duration accuracy** (at true switch positions)

| Model | hi-en | yue-en | es-en | pooled |
|---|---|---|---|---|
| XLM-R | 0.596 | 0.598 | 0.587 | 0.594 |
| Qwen3-0.6B | 0.570 | 0.575 | 0.527 | 0.559 |

Notes: the best epoch (epoch 2 for both models) is chosen on the validation set and there is no separate test set, so scores are slightly optimistic. Each setting is a single run with seed 42.

## Known limitations

- The corpus is synthetic.
- Labels come from heuristics: Hindi data contain Latinized Hindi that the script rule labels as English, and the Spanish scorer is not yet validated against gold labels.
- Units are characters for Han script but words elsewhere, so duration bins are not directly comparable across pairs.
- No multitask vs switch-only comparison yet.

## Roadmap

- Tune hyperparameters, add a held-out test split, repeat runs with several seeds.
- Detect Latinized Hindi and validate Spanish labels (for example against LinCE gold language IDs).
- Compare imbalance handling: weighted cross-entropy, focal and dice losses, tuned thresholds.
- Extend to the remaining language pairs and report per-pair F1 and its spread across pairs.
- Word-by-word streaming demo (next-switch probability and anticipated duration).

## Team

| Member | Contribution |
|---|---|
| Suoang Lu | Labeling: unit segmentation, language ID, switch and duration labels, data pipeline |
| Vivida Siddegowda Mallesh | XLM-R model: prefix input, two heads, training and evaluation |
| Dhanush Murdur Hemanth | Qwen3 model: chunked tokenization, heads, training and evaluation |
| Faizan Ahmed | Baselines (B0, B1, B2), evaluation and result tables, report and documentation |

## References

- P. Xie et al. *SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset.* NeurIPS 2025.
- A. Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020 (XLM-R).
- A. Yang et al. *Qwen3 Technical Report.* arXiv:2505.09388, 2025.
- T. Solorio and Y. Liu. *Learning to Predict Code-Switching Points.* EMNLP 2008.

## License

MIT (see `LICENSE`).
