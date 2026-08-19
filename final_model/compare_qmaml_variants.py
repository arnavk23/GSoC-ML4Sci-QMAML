"""
Head-to-head comparison: this repo's original Q-MAML training loop
(meta/qmaml_loops.py, inner-loop SGD steps + first-order detach trick)
vs. the paper-faithful variant (meta/qmaml_paper_loops.py, Algorithm 1/2
from arXiv:2501.05906, no inner loop during pre-training).

Runs on the real quark-gluon dataset if config.TRAIN_PATH/TEST_PATH point
to actual files; otherwise falls back to synthetic random tensors so the
harness itself can be validated before the dataset is available. Synthetic
results are NOT meaningful physics results -- they only prove the code runs.

Usage:
    python compare_qmaml_variants.py [--epochs N] [--num-qubits N] [--depth N]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--num-qubits", type=int, default=config.NUM_QUBITS)
    p.add_argument("--depth", type=int, default=config.Q_DEPTH)
    p.add_argument("--n-train-tasks", type=int, default=8, help="synthetic mode only")
    p.add_argument("--n-test-tasks", type=int, default=4, help="synthetic mode only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_synthetic_tasks(n_tasks, support=8, query=8, img=125):
    tasks = []
    for _ in range(n_tasks):
        tasks.append({
            "support_X": torch.rand(support, 3, img, img),
            "support_y": torch.randint(0, 2, (support,)),
            "query_X": torch.rand(query, 3, img, img),
            "query_y": torch.randint(0, 2, (query,)),
        })
    return tasks


def try_load_real_tasks():
    """Only works once config.TRAIN_PATH/TEST_PATH point to real .hdf5 files."""
    real = os.path.isfile(config.TRAIN_PATH) and os.path.isfile(config.TEST_PATH)
    if not real:
        return None, None
    from data import train_dataset, test_dataset
    from tasks import generate_meta_tasks
    meta_tasks = generate_meta_tasks(
        train_dataset, config.META_TASK_TYPE, config.META_BIN_COUNT,
        config.SUPPORT_SIZE, config.QUERY_SIZE, config.TG_NUM_TASKS_PER_BIN,
        config.MAX_META_TASKS, config.TG_TRAIN_SEED,
    )
    test_meta_tasks = generate_meta_tasks(
        test_dataset, config.META_TASK_TYPE, config.META_BIN_COUNT,
        config.SUPPORT_SIZE, config.QUERY_SIZE, config.TG_NUM_TASKS_PER_BIN,
        config.MAX_META_TASKS, config.TG_TEST_SEED,
    )
    return meta_tasks, test_meta_tasks


def build_model(num_qubits, depth, seed):
    torch.manual_seed(seed)
    from models import CNNFeatureExtractor, PQCModel, HybridModel, freeze_bn
    cnn = CNNFeatureExtractor(config.CNN_OUTPUT_DIM, num_qubits)
    freeze_bn(cnn)
    pqc = PQCModel(num_qubits, depth, init_type="zero", bound_angles=True)
    return HybridModel(cnn, pqc)


def main():
    args = parse_args()
    config.EPOCHS = args.epochs
    config.NUM_QUBITS = args.num_qubits
    config.Q_DEPTH = args.depth
    config.CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "ckpt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    meta_tasks, test_meta_tasks = try_load_real_tasks()
    using_real = meta_tasks is not None
    if not using_real:
        print("[compare] No real dataset found at config.TRAIN_PATH/TEST_PATH -- "
              "using SYNTHETIC random tensors. Results below only validate that "
              "the two training loops run correctly; they carry no physics meaning.")
        meta_tasks = build_synthetic_tasks(args.n_train_tasks)
        test_meta_tasks = build_synthetic_tasks(args.n_test_tasks)
    else:
        print(f"[compare] Loaded real data: {len(meta_tasks)} train tasks, "
              f"{len(test_meta_tasks)} test tasks.")

    from meta import outer_loop_qmaml, pretrain_learner_paper

    print("\n=== Variant 1: existing outer_loop_qmaml (inner-loop + FOMAML trick) ===")
    model_a = build_model(args.num_qubits, args.depth, args.seed)
    results_a = outer_loop_qmaml(model_a, meta_tasks, test_meta_tasks, config.OUTER_LR, True,
                                  ckpt_name="best_qmaml_existing.pth")

    print("\n=== Variant 2: paper-faithful pretrain_learner_paper (no inner loop) ===")
    model_b = build_model(args.num_qubits, args.depth, args.seed)
    results_b = pretrain_learner_paper(model_b, meta_tasks, test_meta_tasks, config.OUTER_LR, True,
                                        ckpt_name="best_qmaml_paper.pth")

    results = {"qmaml_existing": results_a, "qmaml_paper": results_b}
    with open(os.path.join(RESULTS_DIR, "comparison_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o if not isinstance(o, dict) else o)

    epochs = range(1, args.epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("meta_loss", "Meta-loss (lower is faster convergence)"),
        ("gradient_norms", "||grad||₂ (moderate is best per the paper's discussion)"),
        ("val_accuracy", "Validation accuracy"),
        ("validation_loss", "Validation loss"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, res in results.items():
            ax.plot(epochs, res[key], marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Q-MAML variant comparison ({'real' if using_real else 'SYNTHETIC'} data)")
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "qmaml_variant_comparison.png")
    fig.savefig(out_png, dpi=150)
    print(f"\n[compare] Saved plot to {out_png}")
    print(f"[compare] Saved raw metrics to {os.path.join(RESULTS_DIR, 'comparison_metrics.json')}")


if __name__ == "__main__":
    main()
