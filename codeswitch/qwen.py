"""Qwen3 (decoder-only) variant: one sentence per sample, labels on the last token of each chunk.

The causal mask means a single forward pass yields predictions for every stopping point.
Labelled positions match the XLM-R examples one-to-one, so `score` results are directly comparable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from . import config as C
from .data import make_examples
from .lid import find_units, langs_of, PIECE_RE
from .metrics import score_by_pair
from .trainer import fit
from .utils import DEVICE, sqrt_balanced_weights


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.QWEN_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def split_chunks(text):
    """Return (units, stops); each stop is (char end, index of the current unit).
    Stops are the end of each unit plus every punctuation mark / digit run after it (same as make_examples)."""
    units = find_units(text)
    stops = []
    for i, u in enumerate(units):
        stops.append((u.end(), i))
        gap_end = units[i + 1].start() if i + 1 < len(units) else len(text)
        gap = text[u.end():gap_end]
        stops += [(u.end() + m.end(), i) for m in PIECE_RE.finditer(gap)]
    return units, stops


def build_sample(text, tokenizer, pair=None):
    units, stops = split_chunks(text)
    n = len(units)
    langs = langs_of(units, pair)
    run_len = [1] * n
    for j in range(n - 2, -1, -1):
        if langs[j] == langs[j + 1]:
            run_len[j] = run_len[j + 1] + 1

    input_ids, switch_labels, dur_labels = [], [], []
    prev = 0
    for end, i in stops:
        # Tokenize chunk by chunk so every stopping point lands on a token boundary
        ids = tokenizer(text[prev:end], add_special_tokens=False)["input_ids"]
        prev = end
        if not ids:
            continue
        has_next = i + 1 < n                                    # the last unit has no "next", so no label
        if has_next:
            switch = int(langs[i] != langs[i + 1])
            dur = C.dur_class(run_len[i + 1]) if switch else 0
        else:
            switch = dur = C.IGNORE
        input_ids += ids
        switch_labels += [C.IGNORE] * (len(ids) - 1) + [switch]  # label sits on the chunk's last token
        dur_labels    += [C.IGNORE] * (len(ids) - 1) + [dur]
    return {"input_ids": input_ids, "switch_labels": switch_labels, "dur_labels": dur_labels}


class QwenDataset(Dataset):
    def __init__(self, recs, tokenizer):
        self.items, self.n_truncated = [], 0
        for r in recs:
            smp = build_sample(r["text"], tokenizer, r["pair"])
            if any(l != C.IGNORE for l in smp["switch_labels"]):    # needs at least two units
                self.n_truncated += len(smp["input_ids"]) > C.QWEN_MAX_LEN
                it = {k: v[:C.QWEN_MAX_LEN] for k, v in smp.items()}  # right-truncation keeps earlier positions intact
                it["pair"] = r["pair"]
                self.items.append(it)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_collate(tokenizer):
    def collate(batch):                                              # right padding; labels padded with IGNORE
        L = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: xs + [v] * (L - len(xs))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tokenizer.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "switch_labels": torch.tensor([pad(b["switch_labels"], C.IGNORE) for b in batch]),
            "dur_labels": torch.tensor([pad(b["dur_labels"], C.IGNORE) for b in batch]),
            "pair_ids": torch.tensor([C.PAIR2ID[b["pair"]] for b in batch]),   # evaluation only, not fed to the model
        }
    return collate


def check_alignment(recs, tokenizer):
    """Assert that Qwen labelled positions equal the XLM-R examples one-to-one."""
    for r in recs:
        smp = build_sample(r["text"], tokenizer, r["pair"])
        q = [(a, b) for a, b in zip(smp["switch_labels"], smp["dur_labels"]) if a != C.IGNORE]
        x = [(e["switch"], e["duration"]) for e in make_examples(r["text"], r["pair"])]
        assert q == x, r["text"]


def show_alignment(recs, tokenizer):
    """Print tokens with their labels for one sentence that contains a switch."""
    demo = next(r for r in recs if any(e["switch"] for e in make_examples(r["text"], r["pair"])))
    smp = build_sample(demo["text"], tokenizer, demo["pair"])
    print(demo["pair"], "|", demo["text"])
    for t, a, b in zip(tokenizer.convert_ids_to_tokens(smp["input_ids"]), smp["switch_labels"], smp["dur_labels"]):
        print(f"{t!r:14} switch={a:4} dur={b:4}")


class QwenSwitchDur(nn.Module):
    def __init__(self, name, n_dur, switch_weights):
        super().__init__()
        self.decoder = AutoModel.from_pretrained(name).float()
        hidden = self.decoder.config.hidden_size
        self.drop = nn.Dropout(0.2)
        self.switch_head = nn.Linear(hidden, 2)
        self.dur_head = nn.Linear(hidden, n_dur)
        self.register_buffer("switch_weights", switch_weights)
        self.register_buffer("dur_idx", torch.arange(n_dur, dtype=torch.float))

    def forward(self, input_ids, attention_mask, switch_labels=None, dur_labels=None):
        h = self.drop(self.decoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        switch_logits, dur_logits = self.switch_head(h), self.dur_head(h)
        out = {"switch_logits": switch_logits, "dur_logits": dur_logits}
        if switch_labels is not None:
            l_sw = switch_logits.sum() * 0.0
            if (switch_labels != C.IGNORE).any():
                l_sw = l_sw + F.cross_entropy(switch_logits.flatten(0, 1), switch_labels.flatten(),
                                              weight=self.switch_weights, ignore_index=C.IGNORE)
            l_du = switch_logits.sum() * 0.0
            m = dur_labels != C.IGNORE        # no-switch dur = 0
            if m.any():
                dl, y = dur_logits[m], dur_labels[m]
                l_du = l_du + C.LAMBDA_DUR * F.cross_entropy(dl, y, label_smoothing=C.DUR_SMOOTH)
                s = y > 0                      # ordinal only works when switch = 1
                if s.any():
                    p = dl[s][:, 1:].softmax(-1)               # ignore class 0
                    exp = (p * self.dur_idx[1:]).sum(-1)       # exp = 1,2,3
                    l_du = l_du + C.LAMBDA_ORD * (exp - y[s].float()).abs().mean()
            out["loss"] = l_sw + l_du
            out["loss_sw"], out["loss_du"] = l_sw.detach(), l_du.detach()
        return out


def build(train_recs, val_recs):
    """Return (tokenizer, model, train_dl, val_dl, train_ds)."""
    tok = load_tokenizer()
    check_alignment(val_recs, tok)
    train_ds, val_ds = QwenDataset(train_recs, tok), QwenDataset(val_recs, tok)
    print(f"{train_ds.n_truncated} / {len(train_recs)} training sentences exceed QWEN_MAX_LEN={C.QWEN_MAX_LEN} tokens (tail labels dropped)")
    collate = make_collate(tok)
    train_dl = DataLoader(train_ds, batch_size=C.QWEN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=16, collate_fn=collate)

    sw = [l for it in train_ds.items for l in it["switch_labels"] if l != C.IGNORE]
    weights = sqrt_balanced_weights(sw)
    print("switch class weights:", weights.tolist())
    model = QwenSwitchDur(C.QWEN_NAME, C.N_DUR, weights).to(DEVICE)
    # Freeze the ~150k-token embedding: saves ~1.2 GB of Adam state and reduces overfitting
    model.decoder.embed_tokens.weight.requires_grad_(False)
    return tok, model, train_dl, val_dl, train_ds


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    sw_pred, sw_gold, du_pred, du_gold, du_med, pairs = [], [], [], [], [], []
    tot, nb = np.zeros(3), 0
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pid = batch.pop("pair_ids")[:, None].expand_as(batch["switch_labels"])
        out = model(**batch)
        tot += [out["loss"].item(), out["loss_sw"].item(), out["loss_du"].item()]; nb += 1
        m = batch["switch_labels"] != C.IGNORE
        sw_pred += out["switch_logits"].argmax(-1)[m].tolist(); sw_gold += batch["switch_labels"][m].tolist()
        dl_ = out["dur_logits"][m]
        du_pred += (dl_[:, 1:].argmax(-1) + 1).tolist(); du_gold += batch["dur_labels"][m].tolist()
        du_med  += ((dl_[:, 1:].softmax(-1).cumsum(-1) >= 0.5).int().argmax(-1) + 1).tolist()
        pairs += [C.PAIR_TAGS[i] for i in pid[m].tolist()]
    res = score_by_pair(sw_pred, sw_gold, du_pred, du_gold, pairs)
    g, med = np.array(du_gold), np.array(du_med)
    mk = g > 0
    res["ALL"]["dur_MAE_median"] = round(float(np.abs(med[mk] - g[mk]).mean()), 3)
    res["ALL"].update(val_loss=round(tot[0] / nb, 4), val_loss_cs=round(tot[1] / nb, 4), val_loss_dur=round(tot[2] / nb, 4))
        
    return res


def train(model, train_dl, val_dl):
    """Trains and restores the best epoch (by switch F1 and duration MAE)."""
    return fit(model, train_dl, val_dl, evaluate, C.QWEN_EPOCHS, C.QWEN_LR, C.QWEN_HEAD_LR,
               log_every=100, keep_best=True, tag="Qwen")


def predict(model, tok, text):
    units, stops = split_chunks(text)
    if not units:
        return None
    ids, prev = [], 0
    for end, _ in stops:                                      # same chunked tokenization as training
        ids += tok(text[prev:end], add_special_tokens=False)["input_ids"]
        prev = end
    ids = ids[-C.QWEN_MAX_LEN:]                               # at inference, drop the oldest tokens
    x = torch.tensor([ids], device=DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=torch.ones_like(x))
    p_dur = out["dur_logits"][0, -1].softmax(-1)              # read the last token's output
    return {
        "p_switch": round(out["switch_logits"][0, -1].softmax(-1)[1].item(), 3),
        "duration_if_switch": p_dur[1:].argmax().item() + 1,
        "p_duration": [round(x, 3) for x in p_dur.tolist()],
    }
