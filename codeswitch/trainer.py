"""Model-agnostic training loop shared by XLM-R and Qwen3."""
import torch
from transformers import get_linear_schedule_with_warmup

from . import config as C
from .metrics import print_scores
from .utils import DEVICE, check_mem, free, mem

HEAD_PREFIX = ("switch_head", "dur_head")


def select_score(res):
    """Model-selection score on the pooled validation set (higher is better)."""
    return res["ALL"]["switch_F1"] - 0.05 * res["ALL"]["dur_MAE"]


def fit(model, train_dl, val_dl, evaluate_fn, epochs, lr, head_lr,
        log_every=100, keep_best=False, weight_decay=0.01, tag="model"):
    """Train with separate LRs for the backbone and the two heads.

    The batch must contain `pair_ids` (used only for evaluation, popped before the forward pass).
    Parameters with requires_grad=False (e.g. a frozen embedding) are skipped automatically.
    Returns the per-epoch validation results.
    """
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(HEAD_PREFIX)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": head_lr}],
                            weight_decay=weight_decay)
    total_steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    history, best, best_state = [], float("-inf"), None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch.pop("pair_ids")
            loss = model(**batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if step % log_every == 0:
                check_mem()
                if step == 0 and epoch == 0:
                    mem(f"{tag} step 0")
        print(f"epoch {epoch + 1}: train loss {total / len(train_dl):.4f}")
        res = evaluate_fn(model, val_dl)
        print_scores("  val", res)
        history.append(res)
        if keep_best and select_score(res) > best:
            best = select_score(res)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  ^ best so far")
        free(); mem(f"{tag} epoch {epoch + 1}")

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    del opt, sched, best_state
    free()
    return history
