"""XLM-R (encoder) variant: one example per prefix, `<mask>` appended, heads read the `<mask>` hidden state."""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from . import config as C
from .lid import find_units
from .metrics import score_by_pair
from .trainer import fit
from .utils import DEVICE, sqrt_balanced_weights


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.XLMR_NAME)
    tok.truncation_side = "left"      # when too long, drop the oldest context and keep what is near the prediction point
    return tok


class SwitchDurDataset(Dataset):
    def __init__(self, raw, tokenizer):
        self.items = []
        for e in raw:
            # The prediction slot is a <mask> token appended to the prefix (same as at inference time)
            enc = tokenizer(e["text"] + tokenizer.mask_token, truncation=True, max_length=C.MAX_LEN)
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "switch_labels": e["switch"],
                "dur_labels": e["duration"],
                "pair": e["pair"],
            })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class BucketBatchSampler:
    """Group similar-length examples into a batch to cut padding. Train: shuffled chunks, batch order shuffled.
    Val (shuffle=False): globally sorted by length."""
    def __init__(self, lens, bs, mult=C.BUCKET_MULT, shuffle=True):
        self.lens, self.bs, self.mult, self.shuffle = lens, bs, mult, shuffle

    def __len__(self):
        return math.ceil(len(self.lens) / self.bs)

    def __iter__(self):
        n = len(self.lens)
        idx = np.random.permutation(n) if self.shuffle else np.arange(n)
        step = self.bs * self.mult if self.shuffle else n
        batches = []
        for i in range(0, n, step):
            ch = sorted(idx[i:i + step], key=lambda j: self.lens[j])
            batches += [ch[k:k + self.bs] for k in range(0, len(ch), self.bs)]
        if self.shuffle:
            random.shuffle(batches)           # avoid a short-to-long curriculum
        return iter(batches)


def downsample_negatives(raw, keep=C.KEEP_NEG, seed=C.SEED):
    """Keep every switch=1 example and a `keep` fraction of switch=0 examples (training set only)."""
    rng = random.Random(seed)
    return [e for e in raw if e["switch"] == 1 or rng.random() < keep]


def make_collate(tokenizer):
    def collate(batch):
        enc = tokenizer.pad(
            [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
            return_tensors="pt")
        enc["switch_labels"] = torch.tensor([b["switch_labels"] for b in batch])
        enc["dur_labels"] = torch.tensor([b["dur_labels"] for b in batch])
        enc["pair_ids"] = torch.tensor([C.PAIR2ID[b["pair"]] for b in batch])   # per-pair evaluation only, not fed to the model
        return enc
    return collate


class SwitchDurModel(nn.Module):
    def __init__(self, name, n_dur, switch_weights, mask_token_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.switch_head = nn.Linear(hidden, 2)
        self.dur_head = nn.Linear(hidden, n_dur)
        self.mask_token_id = mask_token_id
        self.register_buffer("switch_weights", switch_weights)

    def forward(self, input_ids, attention_mask, switch_labels=None, dur_labels=None):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pos = (input_ids == self.mask_token_id).int().argmax(1)                 # position of <mask>
        h = self.drop(H[torch.arange(H.size(0), device=H.device), pos])
        switch_logits, dur_logits = self.switch_head(h), self.dur_head(h)
        out = {"switch_logits": switch_logits, "dur_logits": dur_logits}
        if switch_labels is not None:
            l_sw = F.cross_entropy(switch_logits, switch_labels, weight=self.switch_weights)
            l_du = switch_logits.new_zeros(())
            if (dur_labels != C.IGNORE).any():
                l_du = C.LAMBDA_DUR * F.cross_entropy(dur_logits, dur_labels, ignore_index=C.IGNORE)
            out["loss"] = l_sw + l_du
            out["loss_sw"], out["loss_du"] = l_sw.detach(), l_du.detach()
        return out


def build(train_raw, val_raw):
    """Return (tokenizer, model, train_dl, val_dl). Training data is negative-downsampled (see C.KEEP_NEG);
    class weights are computed on the downsampled set, validation stays untouched."""
    tok = load_tokenizer()
    collate = make_collate(tok)
    train_ds_raw = downsample_negatives(train_raw)
    sw = [e["switch"] for e in train_ds_raw]
    print(f"downsampling: {len(train_raw)} -> {len(train_ds_raw)}, switch=1 share {np.mean(sw):.1%}")

    train_ds, val_ds = SwitchDurDataset(train_ds_raw, tok), SwitchDurDataset(val_raw, tok)
    train_lens = [len(x["input_ids"]) for x in train_ds.items]
    val_lens = [len(x["input_ids"]) for x in val_ds.items]
    train_dl = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_lens, C.BATCH_SIZE), collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_lens, C.VAL_BATCH_SIZE, shuffle=False), collate_fn=collate)

    weights = sqrt_balanced_weights(sw)
    print("switch class weights:", weights.tolist())
    model = SwitchDurModel(C.XLMR_NAME, C.N_DUR, weights, tok.mask_token_id).to(DEVICE)
    return tok, model, train_dl, val_dl


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    sw_pred, sw_gold, du_pred, du_gold, pairs = [], [], [], [], []
    tot, nb = np.zeros(3), 0
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pairs += [C.PAIR_TAGS[i] for i in batch.pop("pair_ids").tolist()]
        out = model(**batch)
        tot += [out["loss"].item(), out["loss_sw"].item(), out["loss_du"].item()]; nb += 1
        sw_pred += out["switch_logits"].argmax(-1).tolist(); sw_gold += batch["switch_labels"].tolist()
        du_pred += (out["dur_logits"][:, 1:].argmax(-1) + 1).tolist();    du_gold += batch["dur_labels"].tolist()
    res = score_by_pair(sw_pred, sw_gold, du_pred, du_gold, pairs)
    res["ALL"].update(val_loss=round(tot[0] / nb, 4), val_loss_cs=round(tot[1] / nb, 4), val_loss_dur=round(tot[2] / nb, 4))
    return res

def train(model, train_dl, val_dl):
    return fit(model, train_dl, val_dl, evaluate, C.EPOCHS, C.LR, C.HEAD_LR, log_every=500, keep_best=True, tag="XLM-R")


def predict(model, tok, text):
    """Predict from the text typed so far."""
    if not find_units(text):
        return None
    inputs = tok(text.rstrip() + tok.mask_token, return_tensors="pt", truncation=True, max_length=C.MAX_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    p_dur = out["dur_logits"].softmax(-1)[0]
    return {
        "p_switch": round(out["switch_logits"].softmax(-1)[0, 1].item(), 3),
        "duration_if_switch": p_dur[1:].argmax().item() + 1,     # MAX_DUR means ">= MAX_DUR"; meaningful only if a switch happens
        "p_duration": [round(x, 3) for x in p_dur.tolist()],
    }
