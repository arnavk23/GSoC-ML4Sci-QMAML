"""
Paper-faithful Q-MAML (arXiv:2501.05906, Lee/Cho/Kim), adapted to few-shot
HEP classification instead of the paper's VQE/Hamiltonian setting.

Key difference from meta/qmaml_loops.py (this repo's original outer_loop_qmaml):
  - qmaml_loops.py runs K inner-loop SGD steps on theta *before* the outer
    query loss, then uses a first-order trick, w_query = (wT - w0).detach() + w0,
    so the outer gradient still only sees d(w_query)/d(w0) = I. That's a
    reasonable FOMAML-style hybrid, but it isn't what the paper describes.
  - The paper's Algorithm 1 has *no* inner-loop adaptation during pre-training
    at all: the Learner's output w0 = h_W(phi) is plugged directly into the
    cost function, and the meta-objective argmin_W sum_i l_Ti(g(h_W(phi_i)))
    is backpropagated straight through (Eq. 2-3). Task-specific adaptation
    (Algorithm 2) only happens afterwards, with the Learner frozen, purely to
    report how fast a fresh PQC converges from that initialization.

This module keeps those two phases separate, mirroring the paper 1:1:
  - pretrain_learner_paper()  == Algorithm 1
  - adapt_to_task_paper()     == Algorithm 2 (used only for eval, per task)
"""
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from config import config
from models.learner import Learner, compute_task_embedding
from models.cnn_extractor import set_cnn_trainable, freeze_bn


# Algorithm 2: Adaptation to a single task (Learner frozen, PQC-only)

def adapt_to_task_paper(
    model: nn.Module,
    w0: torch.Tensor,
    support_X: torch.Tensor,
    support_y: torch.Tensor,
    steps: int,
    lr: float,
) -> torch.Tensor:
    """Fine-tune PQC weights starting from the Learner's w0, on `support_X/y` only.
    No Learner update happens here — this is pure Algorithm 2."""
    with torch.no_grad():
        support_feats = model.cnn(support_X)

    theta = nn.Parameter(w0.clone().detach().to(torch.float64), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = model.pqc(support_feats, weights_override=theta)
        loss = loss_fn(logits, support_y)
        loss.backward()
        opt.step()

    return theta.detach()


# Algorithm 1: Pre-training phase (Learner-only, no inner loop)

def pretrain_learner_paper(
    model: nn.Module,
    meta_tasks: List[Dict[str, Any]],
    test_meta_tasks: List[Dict[str, Any]],
    outer_lr: float,
    eval_metrics: bool,
    ckpt_name: str = "best_qmaml_paper_learner.pth",
) -> Dict[str, List[float]]:

    loss_fn = nn.CrossEntropyLoss()

    pqc_shape = tuple(model.pqc.weights.shape)
    learner = Learner(config.CNN_OUTPUT_DIM, pqc_shape, config.LEARNER_HIDDEN).double()
    learner_dtype = next(learner.parameters()).dtype

    if config.FREEZE_CNN_DURING_META:
        try:
            set_cnn_trainable(model.cnn, train_layer4=True, train_to_angles=True)
            freeze_bn(model.cnn)
        except Exception:
            for p in model.cnn.backbone.parameters():
                p.requires_grad = False
            for name, p in model.cnn.backbone.named_parameters():
                if "layer4" in name and "bn" not in name:
                    p.requires_grad = True
            for p in model.cnn.to_angles.parameters():
                p.requires_grad = True
            freeze_bn(model.cnn)
        model.cnn.eval()

    cnn_fast_params = [p for n, p in model.cnn.named_parameters()
                       if p.requires_grad and ("to_angles" in n or "layer4" in n)]

    outer_opt = torch.optim.Adam(
        [
            {"params": learner.parameters(),      "lr": outer_lr},
            {"params": model.pqc.fc.parameters(), "lr": outer_lr},
            {"params": cnn_fast_params,           "lr": outer_lr * 0.2},
        ]
    )

    meta_loss_hist, val_loss_hist, grad_hist = [], [], []
    metrics = {"train_accuracy": [], "val_accuracy": [], "precision": [], "recall": [], "f1_score": []}
    best_acc = 0.0

    def _grad_norm_preclip(*param_iters) -> float:
        s = 0.0
        for it in param_iters:
            for p in it:
                if p.grad is not None:
                    s += float(p.grad.detach().pow(2).sum().item())
        return s ** 0.5

    for epoch in range(config.EPOCHS):
        model.train()
        meta_loss_epoch, epoch_grad_norms, train_accs = 0.0, [], []

        for task in meta_tasks:
            outer_opt.zero_grad()

            # No inner loop: w0 IS the parameter evaluated for the meta-objective (Eq. 2)
            emb = compute_task_embedding(model.cnn, task["support_X"]).to(learner_dtype)
            w0 = config.W0_SCALE * learner(emb)

            support_feats = model.cnn(task["support_X"])
            logits = model.pqc(support_feats, weights_override=w0)
            loss = loss_fn(logits, task["support_y"])
            loss.backward()

            gn = _grad_norm_preclip(learner.parameters(), model.pqc.fc.parameters(), cnn_fast_params)
            epoch_grad_norms.append(gn)

            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.pqc.fc.parameters(), 1.0)
            if cnn_fast_params:
                torch.nn.utils.clip_grad_norm_(cnn_fast_params, 1.0)

            outer_opt.step()
            meta_loss_epoch += loss.item()

            if eval_metrics:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    train_accs.append((preds == task["support_y"]).float().mean().item())

        # Validation: Algorithm 2 adaptation on support, evaluate on query (Learner frozen)
        model.eval()
        vloss, val_accs, val_precs, val_recs, val_f1s = 0.0, [], [], [], []
        for t in test_meta_tasks:
            with torch.no_grad():
                emb = compute_task_embedding(model.cnn, t["support_X"]).to(learner_dtype)
                w0 = config.W0_SCALE * learner(emb)
            theta_star = adapt_to_task_paper(
                model, w0, t["support_X"], t["support_y"], config.INNER_STEPS, config.INNER_LR
            )
            with torch.no_grad():
                q_feats = model.cnn(t["query_X"])
                logits = model.pqc(q_feats, weights_override=theta_star)
                vloss += loss_fn(logits, t["query_y"]).item()
                if eval_metrics:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    preds = torch.argmax(logits, dim=1)
                    val_accs.append((preds == t["query_y"]).float().mean().item())
                    val_precs.append(precision_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))
                    val_recs.append(recall_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))
                    val_f1s.append(f1_score(t["query_y"].cpu(), preds.cpu(), zero_division=0))

        n_tasks = max(1, len(meta_tasks))
        meta_loss_hist.append(meta_loss_epoch / n_tasks)
        val_loss_hist.append(vloss / max(1, len(test_meta_tasks)))
        grad_hist.append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0)

        if eval_metrics:
            metrics["train_accuracy"].append(float(np.mean(train_accs)) if train_accs else 0.0)
            metrics["val_accuracy"].append(float(np.mean(val_accs)) if val_accs else 0.0)
            metrics["precision"].append(float(np.mean(val_precs)) if val_precs else 0.0)
            metrics["recall"].append(float(np.mean(val_recs)) if val_recs else 0.0)
            metrics["f1_score"].append(float(np.mean(val_f1s)) if val_f1s else 0.0)
        else:
            for k in metrics:
                metrics[k].append(0.0)

        avg_val_acc = metrics["val_accuracy"][-1]
        if config.SAVE_BEST_MODEL and avg_val_acc > best_acc:
            best_acc = avg_val_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(learner.state_dict(), os.path.join(config.CHECKPOINT_DIR, ckpt_name))

        print(f"[Q-MAML-paper] Epoch {epoch+1}/{config.EPOCHS} | "
              f"Meta-loss {meta_loss_hist[-1]:.4f} | "
              f"Val Loss {val_loss_hist[-1]:.4f} | "
              f"Train Acc {metrics['train_accuracy'][-1]:.4f} | "
              f"Val Acc {metrics['val_accuracy'][-1]:.4f}")

    return {
        "meta_loss": meta_loss_hist,
        "training_loss": meta_loss_hist,
        "validation_loss": val_loss_hist,
        "gradient_norms": grad_hist,
        "train_pvar": [0.0] * config.EPOCHS,
        "val_pvar": [0.0] * config.EPOCHS,
        **metrics,
        "test_metrics": {"accuracy": [], "precision": [], "recall": [], "f1_score": []},
    }


__all__ = ["pretrain_learner_paper", "adapt_to_task_paper"]
