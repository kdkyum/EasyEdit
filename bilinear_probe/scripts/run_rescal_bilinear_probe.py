#!/usr/bin/env python3
"""CLI evaluation script for layer-wise latent relation reasoning.

Core pipeline:
1. Load train and test JSON datasets plus a embeddings ``.pt`` file containing an ``entities`` mapping.
2. Build entity indices and per-relation adjacency matrices for train and test sets (currently a single implicit relation).
3. For each transformer layer: align entity embeddings for train and test, fit a RESCAL relation matrix R on the full train adjacency, then evaluate metrics on both train and test.
4. Record per-layer metrics: Train MSE, Test MSE, and Test Accuracy; save CSV + plots.

Example usage:
        python bilinear_probe/scripts/run_rescal_bilinear_probe.py \
                --train-dataset data/counterfact_city-country.json \
                --test-dataset data/counterfact_city-country_test.json \
                --embeddings hidden_repr/SOCRATES_v1_Mistral-Large-Instruct-2407_embeddings.pt \
                --outdir outputs/cli_metrics_run1
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Headless plotting first
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
 

# Project utils
# Note: previously appended project path and imported data_utils, but it's unused here.


# (Legacy node-splitting + worker code removed after refactor to explicit train/test datasets.)


# ---------- Helpers ----------

def extract_entity_embeddings(raw: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[int, torch.Tensor]]:
    """Extract per-layer embeddings for each entity.

    Parameters
    ----------
    raw : Dict[str, Dict[str, Any]]
        Mapping from entity id to a dict whose keys include layer identifiers (e.g. ``"layer_12"`` or ``"12"``)
        and whose values are tensors.

    Returns
    -------
    Dict[str, Dict[int, torch.Tensor]]
        ``entity_id -> {layer_index -> 1D embedding tensor}``
    """
    out: Dict[str, Dict[int, torch.Tensor]] = {k: {} for k in raw.keys()}
    for eid, val in raw.items():
        for n, v in val.items():
            try:
                layer_idx = int(str(n).split("_")[-1])
            except Exception:
                continue
            out[eid][layer_idx] = v
    return out


def update_Mr_exact_svd(A: torch.Tensor, X_r: torch.Tensor, lambda_R: float) -> torch.Tensor:
    """Closed-form RESCAL relation matrix update using SVD + ridge.

    Solves: ``argmin_R ||A R A^T - X_r||_F^2 + lambda_R * ||R||_F^2`` via projection
    into singular vector basis of ``A``.

    Parameters
    ----------
    A : torch.Tensor
        Entity embedding matrix of shape (n, d).
    X_r : torch.Tensor
        Relation adjacency matrix of shape (n, n).
    lambda_R : float
        Ridge regularization strength.

    Returns
    -------
    torch.Tensor
        Learned relation matrix ``R`` of shape (d, d).
    """
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh.T
    s2 = torch.outer(s, s)
    filt = s2 / (s2.pow(2) + lambda_R)
    C = U.T @ X_r @ U
    C_filt = filt * C
    return V @ C_filt @ V.T


def solve_rescal_by_logistic_regression(
    X: torch.Tensor,
    A: torch.Tensor,
    lambda_reg: float = 0.1,
    lr: float = 1e-4,
    epochs: int = 1000
) -> torch.Tensor:
    """
    Learn relation matrix R using Logistic Regression (BCE loss) via Gradient Descent.
    
    Model: P(edge) = sigmoid(A @ R @ A.T)
    Loss: BCE(logits, X) + lambda_reg * ||R||_F^2
    """
    n, d = A.shape
    device = A.device
    
    # Initialize R
    R = torch.nn.Parameter(torch.empty(d, d, device=device))
    torch.nn.init.xavier_uniform_(R)
    
    optimizer = optim.Adam([R], lr=lr)
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = A @ R @ A.T
        loss = criterion(logits, X)
        
        # L2 Regularization
        if lambda_reg > 0:
            l2_reg = torch.norm(R) ** 2
            loss += lambda_reg * l2_reg
        
        loss.backward()
        optimizer.step()
        
    return R.detach()


# Average Precision (PR-AUC)
try:
    from sklearn.metrics import average_precision_score as _sk_average_precision  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def average_precision_safe(y_true_np: np.ndarray, y_score_np: np.ndarray) -> float:
    """Compute Average Precision (PR-AUC) with sklearn if available, else a manual fallback.

    Returns NaN when there are no positive labels or inputs are empty.
    """
    if y_true_np.size == 0 or y_score_np.size == 0:
        return float("nan")
    pos = int(y_true_np.sum())
    if pos == 0:
        return float("nan")
    if _HAS_SKLEARN:
        try:
            return float(_sk_average_precision(y_true_np, y_score_np))
        except Exception:
            pass
    # Manual AP (step-wise interpolation)
    order = np.argsort(-y_score_np)
    y_true_sorted = y_true_np[order].astype(np.int32)
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    denom = tp + fp
    denom[denom == 0] = 1
    precision = tp / denom
    recall = tp / max(pos, 1)
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    ap = float((precision * delta_recall).sum())
    return ap


def safe_metrics_from_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float, float]:
    """Compute accuracy, precision, recall, F1 for binary predictions.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth binary tensor.
    y_pred : torch.Tensor
        Predicted binary tensor (0/1).

    Returns
    -------
    Tuple[float, float, float, float]
        (accuracy, precision, recall, F1) with NaNs where undefined.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    correct = (y_pred == y_true).float().mean().item() if y_true.numel() else float("nan")
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else float("nan")
    return correct, prec, rec, f1


def build_entity_index(data: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str], List[int], List[int]]:
    """
    Build bidirectional indices for entities (subjects and answers) from dataset examples.

    Each example in ``data`` is expected to contain a ``"subject"`` field and an ``"answers"`` field.
    The ``"answers"`` field may be either a single string or a list of strings; in the latter case,
    only the first answer is used for index construction.

    Subjects and answers are:
        - Cast to strings if they are not already strings.
        - Collected into a set of entities, where subjects are marked with flag ``1`` and answers
        with flag ``0``.
        - Assigned integer indices based on the order in the deduplicated entity set.

    Args:
        data: A list of examples, where each example is a dictionary containing at least:
            - ``"subject"``: the subject entity (any type, cast to ``str``).
            - ``"answers"``: either a string or a list of strings. If it is a list,
                the first element is used.

    Returns:
        Tuple containing:
            - entity2idx: A dictionary mapping entity names (``str``) to integer indices.
            - idx2entity: A dictionary mapping integer indices back to entity names (``str``).
            - subject_indices: A list of integer indices corresponding to subject entities.
            - answer_indices: A list of integer indices corresponding to answer entities.
    """
    entity_set = set()
    for item in data:
        subject = item.get("subject")
        answers = item.get("answers")

        if subject is None or answers is None:
            continue

        if isinstance(answers, list):
            if not answers:
                continue
            answer = answers[0]
        else:
            answer = answers

        if not isinstance(subject, str):
            subject = str(subject)
        if not isinstance(answer, str):
            answer = str(answer)

        entity_set.add((1, subject))
        entity_set.add((0, answer))
    entity2idx = {name: i for i, (is_subject, name) in enumerate(entity_set)}
    idx2entity = {idx: name for name, idx in entity2idx.items()}
    subject_indices = [entity2idx[name] for is_subject, name in entity_set if is_subject == 1]
    answer_indices = [entity2idx[name] for is_subject, name in entity_set if is_subject == 0]
    return entity2idx, idx2entity, subject_indices, answer_indices


def build_relation_adjs(dataset: List[Dict[str, Any]], entity2idx: Dict[str, int], relation: str) -> Dict[str, torch.Tensor]:
    """Construct adjacency matrices for each relation.

    Current implementation assumes a single implicit relation ("city-country").
    Returns a dict mapping relation name -> (n,n) binary adjacency tensor.
    """
    rel_to_adj: Dict[str, torch.Tensor] = {}
    num_entities = len(entity2idx)
    adj = torch.zeros((num_entities, num_entities), dtype=torch.float32)

    for item in dataset:
        subject = item.get("subject")
        answers = item.get("answers")
        if subject is None or answers is None:
            continue

        if isinstance(answers, list):
            if not answers:
                continue
            answer = answers[0]
        else:
            answer = answers

        if subject not in entity2idx or answer not in entity2idx:
            continue
        row = entity2idx[subject]
        col = entity2idx[answer]
        adj[row, col] = 1.0
    rel_to_adj[relation] = adj
    return rel_to_adj


def map_subject_case_ids(dataset: List[Dict[str, Any]], entity2idx: Dict[str, int]) -> Dict[int, List[Any]]:
    """Map each subject entity index to the list of case_ids observed in the dataset."""
    mapping: Dict[int, List[Any]] = {}
    for item in dataset:
        subject = item.get("subject")
        case_id = item.get("case_id")
        if subject is None or case_id is None:
            continue
        if not isinstance(subject, str):
            subject = str(subject)
        subj_idx = entity2idx.get(subject)
        if subj_idx is None:
            continue
        mapping.setdefault(subj_idx, []).append(case_id)
    return mapping

def build_aligned_embedding_matrix(v: Dict[str, Dict[int, torch.Tensor]], entity2idx: Dict[str, int]) -> Tuple[List[int], np.ndarray, List[str]]:
    """Discover available layer indices (union across entities) and report missing entities.

    Returns a dummy array for shape reference plus list of entities lacking embeddings.
    """
    layer_set: set[int] = set()
    sample_vec: Optional[torch.Tensor] = None
    for eid, layer_map in v.items():
        for lidx, vec in layer_map.items():
            try:
                layer_set.add(int(lidx))
            except Exception:
                continue
            if sample_vec is None:
                sample_vec = vec
    layers = sorted(layer_set)
    missing = [eid for eid in entity2idx.keys() if eid not in v]
    if sample_vec is None:
        d = 0
    else:
        d = int(sample_vec.shape[-1]) if hasattr(sample_vec, "shape") else int(len(sample_vec))
    n = len(entity2idx)
    return layers, np.zeros((n, d), dtype=np.float32), missing


# ---------- Module-level worker function for multiprocessing ----------

def _compute_layer_worker(
    layer_idx: int,
    v: Dict[str, Dict[int, torch.Tensor]],
    train_entity_order: List[str],
    test_entity_order: List[str],
    relation: str,
    train_X_r: Dict[str, torch.Tensor],
    test_X_r: Dict[str, torch.Tensor],
    test_sub_idx: List[int],
    test_ans_idx: List[int],
    test_subject_case_ids: Dict[int, List[Any]],
    lambda_R: float,
    threshold: float,
    epochs: int,
    device: str,
) -> Dict[str, Dict[str, Any]]:
    """Worker function for computing metrics for a single layer (must be at module level for pickle)."""
    
    def A_for_layer_local(entity_order: List[str], layer_idx_local: int) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for eid in entity_order:
            vec = v.get(eid, {}).get(layer_idx_local, None)
            if vec is None:
                if rows:
                    d = rows[0].shape[-1]
                else:
                    # infer dim from any existing entity that has this layer
                    for veid, layer_map in v.items():
                        if layer_idx_local in layer_map:
                            d = int(layer_map[layer_idx_local].shape[-1])
                            break
                    else:
                        raise ValueError(f"Layer {layer_idx_local} not found in any entity embeddings.")
                rows.append(torch.zeros(d))
            else:
                rows.append(vec.float())
        return torch.stack(rows, dim=0)
    
    device_t = torch.device(device)
    A_train = A_for_layer_local(train_entity_order, layer_idx).to(device_t)
    A_test = A_for_layer_local(test_entity_order, layer_idx).to(device_t)
    
    layer_results: Dict[str, Dict[str, Any]] = {}
    train_adj = train_X_r.get(relation)
    test_adj = test_X_r.get(relation)
    train_adj_t = train_adj.to(device_t)
    
    # Fit R on full train adjacency
    if epochs == -1:
        R = update_Mr_exact_svd(A_train, train_adj_t, lambda_R=lambda_R)
        use_sigmoid = False
    else:
        R = solve_rescal_by_logistic_regression(
            train_adj_t, 
            A_train, 
            lambda_reg=lambda_R,
            epochs=epochs
        )
        use_sigmoid = True

    with torch.no_grad():
        raw_train = A_train @ R @ A_train.T
        pred_train = torch.sigmoid(raw_train) if use_sigmoid else raw_train
        train_mse_val = F.mse_loss(pred_train, train_adj_t).item()
        correct_case_ids: List[Any] = []
        if test_adj is not None:
            test_adj_t = test_adj.to(device_t)
            raw_test = A_test @ R @ A_test.T
            pred_test = torch.sigmoid(raw_test) if use_sigmoid else raw_test
            test_mse_val = F.mse_loss(pred_test, test_adj_t).item()
            if test_sub_idx and test_ans_idx:
                sub_idx_tensor = torch.tensor(test_sub_idx, dtype=torch.long, device=device_t)
                ans_idx_tensor = torch.tensor(test_ans_idx, dtype=torch.long, device=device_t)
                pred_subset = pred_test[sub_idx_tensor][:, ans_idx_tensor]
                true_subset = test_adj_t[sub_idx_tensor][:, ans_idx_tensor]
                y_pred = torch.argmax(pred_subset, dim=1).flatten()
                y_true = torch.argmax(true_subset, dim=1).flatten()
                acc_v = y_pred.eq(y_true).float().mean().item()

                pred_entity_idx = ans_idx_tensor[y_pred]
                true_entity_idx = ans_idx_tensor[y_true]
                correct_mask = pred_entity_idx.eq(true_entity_idx)
                correct_mask_cpu = correct_mask.to("cpu").tolist()
                for subj_idx_val, is_correct in zip(test_sub_idx, correct_mask_cpu):
                    if not is_correct:
                        continue
                    ids = test_subject_case_ids.get(int(subj_idx_val)) or []
                    correct_case_ids.extend(ids)
            else:
                acc_v = float('nan')
        else:
            test_mse_val = float('nan')
            acc_v = float('nan')
    layer_results[relation] = {
        "train_mse": train_mse_val,
        "test_mse": test_mse_val,
        "acc": acc_v,
        "correct_case_ids": correct_case_ids,
    }
    return layer_results


def compute_metrics_for_relation(
    v: Dict[str, Dict[int, torch.Tensor]],
    train_entity2idx: Dict[str, int],
    test_entity2idx: Dict[str, int],
    train_X_r: Dict[str, torch.Tensor],
    test_X_r: Dict[str, torch.Tensor],
    test_sub_idx: List[int],
    test_ans_idx: List[int],
    test_subject_case_ids: Dict[int, List[Any]],
    relation: str,
    lambda_R: float,
    threshold: float,
    epochs: int = 1000,
    device: str = "cpu",
    num_workers: int = 1,
) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[List[Any]]]]:
    """Compute per-layer metrics for each relation using explicit train/test adjacencies.

        For each layer:
            * Align embeddings for train and test entities.
            * Fit RESCAL relation matrix R on the full train adjacency.
            * Evaluate metrics on train (MSE) and test (MSE + Accuracy).

                Returns
                -------
                Tuple containing:
                        - layers: List[int] of actual layer indices (sorted)
                        - train_mse: dict mapping relation -> per-layer Train MSE values
                        - test_mse: dict mapping relation -> per-layer Test MSE values
                        - acc: dict mapping relation -> per-layer Test Accuracy values
                        - correct_ids: dict mapping relation -> per-layer lists of correctly predicted case_ids
    """

    # Layers determined from embedding dict (use union of all entities' layer keys)
    # Reuse helper to get layer list (entity2idx choice irrelevant for layer discovery)
    any_entity2idx = train_entity2idx if train_entity2idx else test_entity2idx
    layers, _, _ = build_aligned_embedding_matrix(v, any_entity2idx)

    train_entity_order = [eid for eid, _ in sorted(train_entity2idx.items(), key=lambda kv: kv[1])]
    test_entity_order = [eid for eid, _ in sorted(test_entity2idx.items(), key=lambda kv: kv[1])]

    # Containers
    train_mse: Dict[str, List[float]] = {relation: []}
    test_mse: Dict[str, List[float]] = {relation: []}
    acc: Dict[str, List[float]] = {relation: []}
    correct_ids: Dict[str, List[List[Any]]] = {relation: []}

    def _store_layer_results(layer_res: Dict[str, Dict[str, Any]]) -> None:
        vals = layer_res.get(relation, {})
        train_mse[relation].append(float(vals.get("train_mse", float("nan"))))
        test_mse[relation].append(float(vals.get("test_mse", float("nan"))))
        acc[relation].append(float(vals.get("acc", float("nan"))))
        correct_ids[relation].append(list(vals.get("correct_case_ids", [])))

    device_t = torch.device(device)
    use_parallel = (num_workers is not None and int(num_workers) > 1 and str(device_t) == 'cpu')

    parallel_complete = False
    if use_parallel and layers:
        actual_workers = min(int(num_workers), len(layers), os.cpu_count() or 1)
        if actual_workers > 1:
            try:
                print(f"Parallelizing across layers with {actual_workers} threads...", flush=True)
                args_list = [
                    (
                        layer_idx,
                        v,
                        train_entity_order,
                        test_entity_order,
                        relation,
                        train_X_r,
                        test_X_r,
                        test_sub_idx,
                        test_ans_idx,
                        test_subject_case_ids,
                        lambda_R,
                        threshold,
                        epochs,
                        device,
                    )
                    for layer_idx in layers
                ]
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = [executor.submit(_compute_layer_worker, *args) for args in args_list]
                    for fut in as_completed(futures):
                        _store_layer_results(fut.result())
                parallel_complete = True
            except Exception as exc:
                print(f"[warn] Parallel evaluation failed ({exc}); falling back to sequential.", file=sys.stderr, flush=True)

    if not parallel_complete:
        for layer_idx in layers:
            layer_res = _compute_layer_worker(
                layer_idx,
                v,
                train_entity_order,
                test_entity_order,
                relation,
                train_X_r,
                test_X_r,
                test_sub_idx,
                test_ans_idx,
                test_subject_case_ids,
                lambda_R,
                threshold,
                epochs,
                device,
            )
            _store_layer_results(layer_res)

    return layers, train_mse, test_mse, acc, correct_ids
    
def plot_and_save_overview(train_mse, test_mse, acc, out_png: str):
    """Generate and save overview plots for Train MSE, Test MSE, and Test Accuracy per layer."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    # Train MSE
    for rel, vals in train_mse.items():
        axes[0].plot(range(len(vals)), vals, label=f"Train MSE ({rel})")
    axes[0].set_title('Train MSE per Layer'); axes[0].set_xlabel('Layer'); axes[0].set_ylabel('MSE'); axes[0].legend(); axes[0].grid()
    # Test MSE
    for rel, vals in test_mse.items():
        axes[1].plot(range(len(vals)), vals, label=f"Test MSE ({rel})")
    axes[1].set_title('Test MSE per Layer'); axes[1].set_xlabel('Layer'); axes[1].set_ylabel('MSE'); axes[1].legend(); axes[1].grid()
    # Accuracy
    for rel, vals in acc.items():
        axes[2].plot(range(len(vals)), vals, label=f"Accuracy ({rel})")
    axes[2].set_title('Test Accuracy per Layer'); axes[2].set_xlabel('Layer'); axes[2].set_ylabel('Accuracy'); axes[2].legend(); axes[2].grid()
    plt.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_and_save_pr_auc(pr_auc, out_png: str):
    """Deprecated: PR-AUC plotting is no longer used (kept for compatibility)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.text(0.5, 0.5, 'PR-AUC not available', ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.tight_layout(); fig.savefig(out_png); plt.close(fig)


def save_metrics_csv(outdir: str, relation: str, layer_indices: List[int], train_mse, test_mse, acc):
    """Persist per-layer Train MSE, Test MSE, and Test Accuracy per relation as CSV files."""
    os.makedirs(outdir, exist_ok=True)
    max_len = max(len(train_mse.get(relation, [])), len(test_mse.get(relation, [])), len(acc.get(relation, [])))
    rows = []
    for i in range(max_len):
        rows.append({
            "layer": layer_indices[i] if i < len(layer_indices) else i,
            "train_mse": train_mse.get(relation, [np.nan]*max_len)[i] if i < len(train_mse.get(relation, [])) else np.nan,
            "test_mse": test_mse.get(relation, [np.nan]*max_len)[i] if i < len(test_mse.get(relation, [])) else np.nan,
            "accuracy": acc.get(relation, [np.nan]*max_len)[i] if i < len(acc.get(relation, [])) else np.nan,
        })
    df_rel = pd.DataFrame(rows)
    df_rel.to_csv(os.path.join(outdir, f"metrics_{relation.replace('/', '_')}.csv"), index=False)


def save_bilinear_correct_ids(outdir: str, layer_indices: List[int], layer_case_ids: List[List[Any]]) -> None:
    """Save per-layer lists of correctly predicted case_ids."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    max_len = max(len(layer_indices), len(layer_case_ids))
    for i in range(max_len):
        layer_val = layer_indices[i] if i < len(layer_indices) else i
        case_ids = layer_case_ids[i] if i < len(layer_case_ids) else []
        rows.append({"layer": layer_val, "case_ids": case_ids})
    with open(os.path.join(outdir, "bilinear_correct_ids.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# ---------- Main ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate latent relation reasoning from embeddings across layers (CLI)")
    p.add_argument("--train-dataset", type=str, required=True, help="Path to training dataset JSON")
    p.add_argument("--test-dataset", type=str, required=True, help="Path to test dataset JSON")
    p.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .pt file containing 'entities'")
    p.add_argument("--relation", type=str, default="person-city", help="Relation to evaluate (currently implicit)")
    p.add_argument("--lambda-R", dest="lambda_R", type=float, default=0.1, help="Ridge lambda for RESCAL update")
    p.add_argument("--epochs", type=int, default=-1, help="Number of epochs for Logistic Regression training (default -1: use closed-form SVD)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary predictions on test adjacency")
    p.add_argument("--outdir", type=str, default="outputs/cli_metrics", help="Directory to save relation-wise results")
    p.add_argument("--no-plots", action="store_true", help="Disable plot saving")
    p.add_argument("--device", type=str, default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel workers across layers (CPU only, max 4 to avoid file descriptor issues)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: orchestrate loading, computation, and persistence of metrics."""
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading dataset...", flush=True)
    all_data = []
    with open(args.train_dataset, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(args.test_dataset, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_data.extend(test_data)
    all_data.extend(train_data)

    print("Loading embeddings...", flush=True)
    data = torch.load(args.embeddings, map_location="cpu")
    if "entities" not in data:
        print("Error: embeddings file missing 'entities' key.", file=sys.stderr)
        return 2
    train_indices = data["meta"]["correct_indices"]["train"]
    test_indices = data["meta"]["correct_indices"]["test"]

    trainset = [all_data[i] for i in train_indices]
    testset = [all_data[i] for i in test_indices]

    print("Trainc acc.: ", len(trainset) / len(train_data), " [{} / {}]".format(len(trainset), len(train_data)))
    print("Test acc.: ", len(testset) / len(test_data), " [{} / {}]".format(len(testset), len(test_data)))

    v = extract_entity_embeddings(data["entities"])  # entity -> layer -> vec

    print("Building entity index and relation adjacencies...", flush=True)
    train_ent2idx, train_idx2ent, _, _ = build_entity_index(trainset)
    test_ent2idx, test_idx2ent, test_sub_idx, test_ans_idx = build_entity_index(testset)
    test_subject_case_ids = map_subject_case_ids(testset, test_ent2idx)

    train_X_r = build_relation_adjs(trainset, train_ent2idx, args.relation)
    test_X_r = build_relation_adjs(testset, test_ent2idx, args.relation)
    
    rel_to_outdir = {}
    rel_outdir = os.path.join(args.outdir, args.relation)
    os.makedirs(rel_outdir, exist_ok=True)
    rel_to_outdir[args.relation] = rel_outdir

    print(f"\n[Relation] {args.relation} -> {rel_outdir}", flush=True)
    layers, train_mse, test_mse, acc, correct_case_ids = compute_metrics_for_relation(
        v=v,
        train_entity2idx=train_ent2idx,
        test_entity2idx=test_ent2idx,
        train_X_r=train_X_r,
        test_X_r=test_X_r,
        test_sub_idx=test_sub_idx,
        test_ans_idx=test_ans_idx,
        test_subject_case_ids=test_subject_case_ids,
        relation=args.relation,
        lambda_R=args.lambda_R,
        threshold=args.threshold,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
    )

    print("Saving CSV metrics...", flush=True)
    save_metrics_csv(rel_outdir, args.relation, layers, train_mse, test_mse, acc)
    save_bilinear_correct_ids(rel_outdir, layers, correct_case_ids.get(args.relation, []))

    if not args.no_plots:
        print("Saving plots...", flush=True)
        plot_and_save_overview(train_mse, test_mse, acc, out_png=os.path.join(rel_outdir, "metrics_overview.png"))

    # Also dump a quick JSON summary across relations
    summary = {
        "relation": args.relation,
        "outdir": args.outdir,
        "relation_outdirs": rel_to_outdir,
        "lambda_R": args.lambda_R,
        "epochs": args.epochs,
        "lm_train_accuracy": len(trainset) / len(train_data),
        "lm_test_accuracy": len(testset) / len(test_data),
        "n_train_examples": len(trainset),
        "n_test_examples": len(testset),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(args.outdir, "correct_indices.npz"),
        train_indices=np.array(train_indices, dtype=np.int64),
        test_indices=np.array(test_indices, dtype=np.int64),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
