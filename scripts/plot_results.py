#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from glob import glob
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Lazy import for plotting so script can still summarize without matplotlib
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

MetricSpec = Tuple[str, Tuple[str, ...]]  # (label, path elements)

METRICS: List[MetricSpec] = [
    ("rewrite_pre", ("pre", "rewrite_acc")),
    ("rewrite_post", ("post", "rewrite_acc")),
    ("rephrase_pre", ("pre", "rephrase_acc")),
    ("rephrase_post", ("post", "rephrase_acc")),
    ("portability_two-hop_pre", ("pre", "portability", "two_hop_acc")),
    ("portability_two-hop_post", ("post", "portability", "two_hop_acc")),
    ("locality_neighborhood_post", ("post", "locality", "neighborhood_acc")),
]

_LAYER_RE = re.compile(r"layer(\d+)\.json$")


def _get_nested(d: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Any]:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _mean_of_value(v: Any) -> Optional[float]:
    if v is None:
        return None
    # Values appear as lists of floats; support float or list[float]
    if isinstance(v, list):
        flat: List[float] = []
        for x in v:
            if isinstance(x, (int, float)) and not math.isnan(float(x)):
                flat.append(float(x))
        if not flat:
            return None
        return float(mean(flat))
    if isinstance(v, (int, float)) and not math.isnan(float(v)):
        return float(v)
    return None


def summarize_layer_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data: list of cases
    sums: Dict[str, List[float]] = {k: [] for k, _ in METRICS}
    for case in data:
        for label, mpath in METRICS:
            val = _get_nested(case, mpath)
            mv = _mean_of_value(val)
            if mv is not None:
                sums[label].append(mv)
    # average across cases
    return {k: (float(mean(v)) if v else float("nan")) for k, v in sums.items()}


def find_experiments(results_dir: str) -> List[Tuple[str, str]]:
    # returns list of (exp_dir, exp_name) where exp_dir holds layer*.json
    exp_dirs: List[str] = []
    for root, dirs, files in os.walk(results_dir):
        if any(_LAYER_RE.search(fn) for fn in files):
            exp_dirs.append(root)
    exps: List[Tuple[str, str]] = []
    for d in sorted(exp_dirs):
        # name like model/dataset or just dataset
        rel = os.path.relpath(d, results_dir)
        exps.append((d, rel.replace(os.sep, "__")))
    return exps


def collect_layers(exp_dir: str) -> List[Tuple[int, str]]:
    files = glob(os.path.join(exp_dir, "layer*.json"))
    items: List[Tuple[int, str]] = []
    for fp in files:
        m = _LAYER_RE.search(os.path.basename(fp))
        if m:
            items.append((int(m.group(1)), fp))
    return sorted(items, key=lambda x: x[0])


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_metrics(exp_name: str, layers: List[int], summary: Dict[str, List[float]], out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    plt.figure(figsize=(10, 6))
    # group pairs for clarity
    pairs = [
        ("rewrite_pre", "rewrite_post"),
        ("rephrase_pre", "rephrase_post"),
        ("portability_two-hop_pre", "portability_two-hop_post"),
    ]
    colors = {
        "rewrite_pre": "#555555",
        "rewrite_post": "#1f77b4",
        "rephrase_pre": "#999999",
        "rephrase_post": "#ff7f0e",
        "portability_two-hop_pre": "#bbbbbb",
        "portability_two-hop_post": "#2ca02c",
        "locality_neighborhood_post": "#d62728",
    }
    for a, b in pairs:
        if a in summary:
            plt.plot(layers, summary[a], label=a, color=colors.get(a, None), linestyle="--", marker="o", alpha=0.8)
        if b in summary:
            plt.plot(layers, summary[b], label=b, color=colors.get(b, None), linestyle="-", marker="o", alpha=0.9)
    if "locality_neighborhood_post" in summary:
        plt.plot(layers, summary["locality_neighborhood_post"], label="locality_neighborhood_post", color=colors.get("locality_neighborhood_post", None), linestyle=":", marker="s")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(exp_name)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best", fontsize=8)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{exp_name}_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def write_csv(exp_name: str, layers: List[int], summary: Dict[str, List[float]], out_dir: str) -> str:
    ensure_dir(out_dir)
    keys = [k for k, _ in METRICS]
    csv_path = os.path.join(out_dir, f"{exp_name}_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer," + ",".join(keys) + "\n")
        for i, layer in enumerate(layers):
            vals = []
            for k in keys:
                v_list = summary.get(k, [])
                v = v_list[i] if i < len(v_list) else float("nan")
                vals.append("" if (v is None or math.isnan(v)) else f"{v:.6f}")
            f.write(str(layer) + "," + ",".join(vals) + "\n")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="Summarize and plot results across layers")
    ap.add_argument("--results_dir", default="results", help="Root results directory")
    ap.add_argument("--out_dir", default=os.path.join("results", "plots"), help="Directory to save plots/CSVs")
    args = ap.parse_args()

    experiments = find_experiments(args.results_dir)
    if not experiments:
        print(f"No experiments with layer*.json found under {args.results_dir}")
        return

    print(f"Found {len(experiments)} experiments")
    for exp_dir, exp_name in experiments:
        items = collect_layers(exp_dir)
        if not items:
            continue
        layers = [i for i, _ in items]
        per_metric: Dict[str, List[float]] = {k: [] for k, _ in METRICS}
        for layer_id, fp in items:
            summary = summarize_layer_json(fp)
            for k in per_metric.keys():
                per_metric[k].append(summary.get(k, float("nan")))
        png = plot_metrics(exp_name, layers, per_metric, args.out_dir)
        csv = write_csv(exp_name, layers, per_metric, args.out_dir)
        msg = f"Summarized {exp_name}: layers={len(layers)} -> CSV={os.path.relpath(csv)}"
        if png:
            msg += f", PNG={os.path.relpath(png)}"
        print(msg)


if __name__ == "__main__":
    main()
