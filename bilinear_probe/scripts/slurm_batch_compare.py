#!/usr/bin/env python3
"""
Batch runner for scripts/run_rescal_bilinear_probe.py across multiple embedding files,
then aggregate and plot cross-model comparisons.

Usage example:
    python scripts/slurm_batch_compare.py \
        --emb-dir hidden_repr \
        --train-dataset data/counterfact_city-country.json \
        --test-dataset data/counterfact_city-country_test.json \
        --relations city-country \
        --out-root outputs/cli_batch \
        --device cpu

This will:
    1) Find all .pt files under --emb-dir (non-recursive)
    2) Run scripts/run_rescal_bilinear_probe.py once per embedding, saving results under
        {out-root}/{model_name}/
    3) Read the per-relation metrics CSVs and produce combined figures across
        models for Accuracy and Test MSE per layer (if available).
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import subprocess
import datetime
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_embedding_files(emb_dir: str) -> List[str]:
    files = []
    for name in os.listdir(emb_dir):
        path = os.path.join(emb_dir, name)
        if os.path.isfile(path) and name.endswith(".pt"):
            files.append(path)
    return sorted(files)


def model_name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def model_family_from_name(model_name: str) -> str | None:
    """Return model family based on prefix of model_name.

    Allowed families (exact set): {"gpt-oss", "Llama", "Mistral", "OLMo", "Qwen"}.
    Only the beginning of the model name is considered; non-matching models return None.
    """
    s = model_name.strip()
    sl = s.lower()
    # Use substring presence rather than prefix matching
    if "gpt-oss" in sl:
        return "gpt-oss"
    if "llama" in sl:
        return "Llama"
    if "mistral" in sl:
        return "Mistral"
    if "olmo" in sl:
        return "OLMo"
    if "qwen" in sl:
        return "Qwen"
    return None


def run_single_model(
    emb_path: str,
    train_dataset: str,
    test_dataset: str,
    relation: str,
    out_root: str,
    lambda_R: float,
    threshold: float,
    no_plots: bool,
    device: str,
    num_workers: int,
) -> str:
    model_name = model_name_from_path(emb_path)
    outdir = os.path.join(out_root, model_name)
    os.makedirs(outdir, exist_ok=True)

    cli = [
        sys.executable, os.path.join("scripts", "run_rescal_bilinear_probe.py"),
        "--train-dataset", train_dataset,
        "--test-dataset", test_dataset,
        "--embeddings", emb_path,
        "--outdir", outdir,
        "--lambda-R", str(lambda_R),
        "--threshold", str(threshold),
        "--device", device,
        "--num-workers", str(num_workers),
    ]
    # Passing relations is optional; run_rescal_bilinear_probe currently infers from data.
    if relation:
        cli += ["--relation", relation]
    if no_plots:
        cli += ["--no-plots"]

    print(f"\n[RUN] {model_name} -> {outdir}")
    subprocess.run(cli, check=True)
    return outdir


def submit_single_model_slurm(
    emb_path: str,
    train_dataset: str,
    test_dataset: str,
    relation: str,
    out_root: str,
    lambda_R: float,
    threshold: float,
    no_plots: bool,
    device: str,
    num_workers: int,
    slurm_logs: str,
    partition: str | None,
    qos: str | None,
    time: str | None,
    cpus_per_task: int | None,
    gres: str | None,
    job_extra: List[str] | None,
) -> Tuple[str, str]:
    """Create and submit an sbatch job for a single model. Returns (model_name, job_id)."""
    os.makedirs(slurm_logs, exist_ok=True)
    model_name = model_name_from_path(emb_path)
    job_dir = os.path.join(slurm_logs, model_name)
    os.makedirs(job_dir, exist_ok=True)
    outdir = os.path.join(out_root, model_name)
    os.makedirs(outdir, exist_ok=True)
    # Use absolute path for the script
    script_path = os.path.abspath(os.path.join("scripts", "run_rescal_bilinear_probe.py"))
    
    cli = [
        "python3", script_path,
        "--train-dataset", train_dataset,
        "--test-dataset", test_dataset,
        "--embeddings", emb_path,
        "--outdir", outdir,
        "--lambda-R", str(lambda_R),
        "--threshold", str(threshold),
        "--device", device,
        "--num-workers", str(num_workers),
        "--relation", relation,
    ]
    if no_plots:
        cli.append("--no-plots")

    # Build sbatch script
    job_script = os.path.join(job_dir, "slurm_job.sh")
    stdout_path = os.path.join(job_dir, "job.out.%j")
    stderr_path = os.path.join(job_dir, "job.err.%j")
    job_name = f"compare_{model_name}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {stdout_path}",
        f"#SBATCH -e {stderr_path}",
        f"#SBATCH --mem=100000",
    ]
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if time:
        lines.append(f"#SBATCH -t {time}")
    if cpus_per_task:
        lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if job_extra:
        lines += [f"#SBATCH {opt}" for opt in job_extra]

    lines += [
        "",
        "set -euo pipefail",
        f"cd {os.getcwd()}",
        "echo \"[$(date)] Starting job on $(hostname)\"",
        "if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi",
        # User-specified cluster environment setup (CPU only)
        "module purge",
        "module load intel/21.4.0 impi/2021.4",
        "module load python-waterboa/2024.06",
        'eval "$(conda shell.bash hook)"',
        "conda activate reasoning",
        # Avoid BLAS oversubscription when using process-based parallelism
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export TORCH_NUM_THREADS=1",
        "echo \"Running: " + " ".join(cli) + "\"",
        " ".join(cli),
        "echo \"[$(date)] Job finished\"",
    ]

    with open(job_script, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Make script executable
    os.chmod(job_script, 0o755)

    # Submit job
    try:
        res = subprocess.run(["sbatch", job_script], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job for {model_name}:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        print(f"  Script content ({job_script}):")
        with open(job_script, 'r') as f:
            print(f.read())
        raise
    
    # Expected output: "Submitted batch job <id>"
    out = res.stdout.strip()
    job_id = out.split()[-1] if out else "unknown"
    # Save exit status file placeholder
    with open(os.path.join(job_dir, "submitted"), "w") as f:
        f.write(out + "\n")
    return model_name, job_id


def discover_relations_from_outputs(model_outdir: str) -> List[str]:
    summary_path = os.path.join(model_outdir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            return list(summary.get("relations", []))
        except Exception:
            pass
    # Fallback: infer from subfolders
    rels = []
    for name in os.listdir(model_outdir):
        rel_dir = os.path.join(model_outdir, name)
        if os.path.isdir(rel_dir):
            # subdir name is sanitized relation
            # We'll trust that CSV inside matches this name
            csv_path = os.path.join(rel_dir, f"metrics_{name}.csv")
            if os.path.exists(csv_path):
                rels.append(name.replace("_", "/"))
    return sorted(rels)


def aggregate_and_plot(out_root: str, model_outdirs: Dict[str, str], rel: str) -> None:
    combined_dir = os.path.join(out_root, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    models_dir = os.path.join(combined_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    grouped_dir = os.path.join(combined_dir, "grouped")
    os.makedirs(grouped_dir, exist_ok=True)
    models_grouped_dir = os.path.join(models_dir, "grouped")
    os.makedirs(models_grouped_dir, exist_ok=True)

    rel_sanitized = rel.replace("/", "_")
    # Replace prior per-series dicts with full dataframes to keep per-model layer indexing
    metrics_per_model: Dict[str, pd.DataFrame] = {}
    metrics_mse_per_model: Dict[str, pd.DataFrame] = {}

    for model_name, outdir in model_outdirs.items():
        csv_path = os.path.join(outdir, rel_sanitized, f"metrics_{rel_sanitized}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: missing CSV for {model_name} relation {rel}: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: failed reading {csv_path}: {e}")
            continue
        # Validate required columns for accuracy
        missing_cols = [c for c in ["layer", "accuracy"] if c not in df.columns]
        if missing_cols:
            print(f"Warning: CSV {csv_path} missing columns {missing_cols}; skipping accuracy for this model.")
        else:
            metrics_per_model[model_name] = df[["layer", "accuracy"]]
        # Collect optional Test MSE if present
        if {"layer", "test_mse"}.issubset(df.columns):
            metrics_mse_per_model[model_name] = df[["layer", "test_mse"]]

    if not metrics_per_model and not metrics_mse_per_model:
        print(f"No data collected for relation {rel}; skipping plots.")

    # Create figure: Accuracy vs Layer
    if metrics_per_model:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for model_name, dfm in metrics_per_model.items():
            ax.plot(dfm["layer"], dfm["accuracy"], marker='o', label=model_name)
        ax.set_title(f"Accuracy vs Layer ({rel})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(combined_dir, f"compare_{rel_sanitized}.png")
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved comparison figure: {out_png}")

    # Create figure: Test MSE vs Layer
    if metrics_mse_per_model:
        fig_m, ax_m = plt.subplots(1, 1, figsize=(8, 6))
        for model_name, dfm in metrics_mse_per_model.items():
            ax_m.plot(dfm["layer"], dfm["test_mse"], marker='o', label=model_name)
        ax_m.set_title(f"Test MSE vs Layer ({rel})")
        ax_m.set_xlabel("Layer")
        ax_m.set_ylabel("Test MSE")
        ax_m.grid(True)
        ax_m.legend(fontsize=8)
        plt.tight_layout()
        out_png_m = os.path.join(combined_dir, f"compare_{rel_sanitized}_test_mse.png")
        fig_m.savefig(out_png_m)
        plt.close(fig_m)
        print(f"Saved comparison figure: {out_png_m}")

    # Additionally, save per-model-family comparison figures for this relation
    # Build mapping: family -> { model_name -> df(layer, accuracy) }
    fam_map: Dict[str, Dict[str, pd.DataFrame]] = {}
    fam_map_mse: Dict[str, Dict[str, pd.DataFrame]] = {}
    for model_name, dfm in metrics_per_model.items():
        fam = model_family_from_name(model_name)
        if fam is None:
            continue
        fam_map.setdefault(fam, {})[model_name] = dfm
    for model_name, dfm in metrics_mse_per_model.items():
        fam = model_family_from_name(model_name)
        if fam is None:
            continue
        fam_map_mse.setdefault(fam, {})[model_name] = dfm

    for fam, fam_models in sorted(fam_map.items()):
        if not fam_models:
            continue
        fig_f, ax_f = plt.subplots(1, 1, figsize=(8, 6))
        for model_name, dfm in fam_models.items():
            ax_f.plot(dfm["layer"], dfm["accuracy"], marker='o', label=model_name)
        ax_f.set_title(f"Accuracy vs Layer ({rel}) — {fam}")
        ax_f.set_xlabel("Layer")
        ax_f.set_ylabel("Accuracy")
        ax_f.set_ylim(0.0, 1.05)
        ax_f.grid(True)
        ax_f.legend(fontsize=8)

        plt.tight_layout()
        fam_sanitized = fam.replace("/", "_").replace(" ", "_")
        out_png_f = os.path.join(models_dir, f"compare_{rel_sanitized}_{fam_sanitized}.png")
        fig_f.savefig(out_png_f)
        plt.close(fig_f)
        print(f"  ↳ Saved family figure: {out_png_f}")

    for fam, fam_models in sorted(fam_map_mse.items()):
        if not fam_models:
            continue
        fig_fm, ax_fm = plt.subplots(1, 1, figsize=(8, 6))
        for model_name, dfm in fam_models.items():
            ax_fm.plot(dfm["layer"], dfm["test_mse"], marker='o', label=model_name)
        ax_fm.set_title(f"Test MSE vs Layer ({rel}) — {fam}")
        ax_fm.set_xlabel("Layer")
        ax_fm.set_ylabel("Test MSE")
        ax_fm.grid(True)
        ax_fm.legend(fontsize=8)

        plt.tight_layout()
        fam_sanitized = fam.replace("/", "_").replace(" ", "_")
        out_png_fm = os.path.join(models_dir, f"compare_{rel_sanitized}_{fam_sanitized}_test_mse.png")
        fig_fm.savefig(out_png_fm)
        plt.close(fig_fm)
        print(f"  ↳ Saved family figure: {out_png_fm}")

    # Additionally, group plots by normalized e2.type into a single figure with subplots
    # Expected relation format: "e1.type-e2.type"
    # Normalization rules:
    #  - any e2 ending with 'city'    -> base 'city',    subtype = e2 without trailing 'city'
    #  - any e2 ending with 'country' -> base 'country', subtype = e2 without trailing 'country'
    #  - any e2 starting with 'nobel' -> base 'nobel',   subtype = e2 without leading 'nobel'
    #  - any e2 ending with 'year'    -> base 'year',    subtype = e2 without trailing 'year'
    # For these bases, create compare_{base}.png containing one subplot per (e1.type, subtype)
    # with Accuracy vs Layer and lines for each model.
    # Build mapping: base -> { (e1, subtype) -> { model_name -> df(layer, accuracy) } }

    def normalize_e2_type(e2_type: str) -> Tuple[str, str, bool]:
        s = e2_type.lower()
        if s.endswith("city"):
            base = "city"
            subtype = e2_type[: len(e2_type) - 4]
        elif s.endswith("country"):
            base = "country"
            subtype = e2_type[: len(e2_type) - 7]
        elif s.startswith("nobel"):
            base = "nobel"
            subtype = e2_type[5:]
        elif s.endswith("year"):
            base = "year"
            subtype = e2_type[: len(e2_type) - 4]
        else:
            return e2_type, "", False
        subtype = subtype.strip("-_/")
        return base, subtype, True

    e2_groups: Dict[str, Dict[Tuple[str, str], Dict[str, pd.DataFrame]]] = {}
    # family -> base -> (e1, subtype) -> { model_name -> df }
    e2_groups_by_family: Dict[str, Dict[str, Dict[Tuple[str, str], Dict[str, pd.DataFrame]]]] = {}
    
    e1_type, e2_type = rel.split("-", 1)
    base, subtype, normalized = normalize_e2_type(e2_type)
    rel_sanitized = rel.replace("/", "_")

    # Collect dataframes for this relation across models
    metrics_per_model: Dict[str, pd.DataFrame] = {}
    metrics_mse_per_model: Dict[str, pd.DataFrame] = {}
    for model_name, outdir in model_outdirs.items():
        csv_path = os.path.join(outdir, rel_sanitized, f"metrics_{rel_sanitized}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        # Require layer and accuracy for accuracy plots
        if {"layer", "accuracy"}.issubset(df.columns):
            metrics_per_model[model_name] = df[["layer", "accuracy"]]
        # Collect optional Test MSE if present
        if {"layer", "test_mse"}.issubset(df.columns):
            metrics_mse_per_model[model_name] = df[["layer", "test_mse"]]


    if base not in e2_groups:
        e2_groups[base] = {}
    e2_groups[base][(e1_type, subtype)] = metrics_per_model

    # Populate family-specific grouped structures
    for model_name, dfm in metrics_per_model.items():
        fam = model_family_from_name(model_name)
        if fam is None:
            continue
        fam_map = e2_groups_by_family.setdefault(fam, {})
        base_map = fam_map.setdefault(base, {})
        key = (e1_type, subtype)
        rel_map = base_map.setdefault(key, {})
        rel_map[model_name] = dfm

    # Render grouped figures (all models)
    for base, e1_map in sorted(e2_groups.items()):
        if not e1_map:
            continue
        e1_items = sorted(e1_map.items())  # list of ((e1_type, subtype), metrics_per_model)
        n = len(e1_items)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

        for idx, (e1_subtype_key, metrics_per_model) in enumerate(e1_items):
            e1_type, subtype = e1_subtype_key
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            for model_name, dfm in metrics_per_model.items():
                ax.plot(dfm["layer"], dfm["accuracy"], marker='o', label=model_name)
            if subtype:
                ax.set_title(f"{e1_type} -> {base} ({subtype})")
            else:
                ax.set_title(f"{e1_type} -> {base}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True)
            ax.legend(fontsize=7)

        # Test MSE grouped figure (all models) for this base
        if metrics_mse_per_model:
            fig_m, axes_m = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
            for idx, (e1_subtype_key, _) in enumerate(e1_items):
                e1_type, subtype = e1_subtype_key
                r, c = divmod(idx, cols)
                axm = axes_m[r][c]
                for model_name, dfm in metrics_mse_per_model.items():
                    axm.plot(dfm["layer"], dfm["test_mse"], marker='o', label=model_name)
                if subtype:
                    axm.set_title(f"{e1_type} -> {base} ({subtype})")
                else:
                    axm.set_title(f"{e1_type} -> {base}")
                axm.set_xlabel("Layer")
                axm.set_ylabel("Test MSE")
                axm.grid(True)
                axm.legend(fontsize=7)

            for idx in range(n, total_axes):
                r, c = divmod(idx, cols)
                axes_m[r][c].axis('off')

            plt.tight_layout()
            out_png_m = os.path.join(grouped_dir, f"compare_{base}_test_mse.png")
            fig_m.savefig(out_png_m)
            plt.close(fig_m)
            print(f"Saved grouped comparison figure: {out_png_m}")

        # Hide any unused subplots
        total_axes = rows * cols
        for idx in range(n, total_axes):
            r, c = divmod(idx, cols)
            axes[r][c].axis('off')

        plt.tight_layout()
        out_png = os.path.join(grouped_dir, f"compare_{base}.png")
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved grouped comparison figure: {out_png}")

    # Render grouped figures per model family
    for fam, base_map in sorted(e2_groups_by_family.items()):
        fam_sanitized = fam.replace("/", "_").replace(" ", "_")
        for base, e1_map in sorted(base_map.items()):
            if not e1_map:
                continue
            e1_items = sorted(e1_map.items())
            n = len(e1_items)
            cols = 2
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

            for idx, (e1_subtype_key, metrics_per_model) in enumerate(e1_items):
                e1_type, subtype = e1_subtype_key
                r, c = divmod(idx, cols)
                ax = axes[r][c]
                for model_name, dfm in metrics_per_model.items():
                    ax.plot(dfm["layer"], dfm["accuracy"], marker='o', label=model_name)
                if subtype:
                    ax.set_title(f"{e1_type} -> {base} ({subtype}) — {fam}")
                else:
                    ax.set_title(f"{e1_type} -> {base} — {fam}")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0.0, 1.05)
                ax.grid(True)
                ax.legend(fontsize=7)

            # Test MSE grouped figure per family
            fig_m, axes_m = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
            for idx, (e1_subtype_key, metrics_per_model) in enumerate(e1_items):
                e1_type, subtype = e1_subtype_key
                r, c = divmod(idx, cols)
                axm = axes_m[r][c]
                # reuse metrics_per_model to iterate model names; fetch mse df by model
                for model_name in metrics_per_model.keys():
                    dfm = e2_groups_by_family.get(fam, {}).get(base, {}).get((e1_type, subtype), {}).get(model_name)
                    if dfm is None or "test_mse" not in dfm.columns:
                        continue
                    axm.plot(dfm["layer"], dfm["test_mse"], marker='o', label=model_name)
                if subtype:
                    axm.set_title(f"{e1_type} -> {base} ({subtype}) — {fam}")
                else:
                    axm.set_title(f"{e1_type} -> {base} — {fam}")
                axm.set_xlabel("Layer")
                axm.set_ylabel("Test MSE")
                axm.grid(True)
                axm.legend(fontsize=7)

            for idx in range(n, total_axes):
                r, c = divmod(idx, cols)
                axes_m[r][c].axis('off')

            plt.tight_layout()
            out_png_m = os.path.join(models_grouped_dir, f"compare_{base}_{fam_sanitized}_test_mse.png")
            fig_m.savefig(out_png_m)
            plt.close(fig_m)
            print(f"Saved family grouped comparison figure: {out_png_m}")

            # Hide any unused subplots
            total_axes = rows * cols
            for idx in range(n, total_axes):
                r, c = divmod(idx, cols)
                axes[r][c].axis('off')

            plt.tight_layout()
            out_png = os.path.join(models_grouped_dir, f"compare_{base}_{fam_sanitized}.png")
            fig.savefig(out_png)
            plt.close(fig)
            print(f"Saved family grouped comparison figure: {out_png}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch run run_rescal_bilinear_probe.py across embeddings and plot comparisons")
    p.add_argument("--emb-dir", type=str, default="hidden_repr", help="Directory containing *.pt embedding files")
    p.add_argument("--train-dataset", type=str, default="data/counterfact_city-country.json", help="Training dataset JSON path")
    p.add_argument("--test-dataset", type=str, default="data/counterfact_city-country_test.json", help="Test dataset JSON path")
    p.add_argument("--relation", type=str, default="city-country", help="Relation to pass to run_rescal_bilinear_probe (optional)")
    p.add_argument("--out-root", type=str, default="outputs/cli_batch", help="Root directory for model outputs and combined plots")
    p.add_argument("--lambda-R", dest="lambda_R", type=float, default=0.1, help="Ridge lambda for RESCAL update")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary predictions")
    p.add_argument("--no-plots", action="store_true", help="Disable per-model plot saving in run_rescal_bilinear_probe")
    p.add_argument("--device", type=str, default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel workers across layers (passed to run_rescal_bilinear_probe)")
    # SLURM related options
    p.add_argument("--submit-slurm", action="store_true", help="Submit one SLURM job per model (do not run locally)")
    p.add_argument("--slurm-logs", type=str, default="outputs/cli_batch/slurm_logs", help="Directory to store SLURM job scripts and logs")
    p.add_argument("--partition", type=str, default=None, help="SLURM partition")
    p.add_argument("--qos", type=str, default=None, help="SLURM QOS")
    p.add_argument("--time", type=str, default=None, help="SLURM time limit, e.g. 2:00:00")
    p.add_argument("--cpus-per-task", type=int, default=None, help="SLURM CPUs per task")
    p.add_argument("--gres", type=str, default=None, help="SLURM generic resources, e.g. gpu:1")
    p.add_argument("--job-extra", type=str, nargs="*", default=None, help="Additional raw #SBATCH options, e.g. --account=acct --constraint=...")
    p.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing per-model outputs; do not run or submit models")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)

    emb_files = find_embedding_files(args.emb_dir)
    if not emb_files:
        print(f"No .pt embeddings found in {args.emb_dir}")
        return 0
    print(f"Found {len(emb_files)} embeddings:")
    for f in emb_files:
        print(" -", os.path.basename(f))

    # If cpus-per-task is specified and user didn't set num-workers explicitly (>1),
    # default to using that many workers per job for layer-parallel CPU processing.
    if args.num_workers == 1 and args.cpus_per_task:
        effective_num_workers = args.cpus_per_task
    else:
        effective_num_workers = args.num_workers

    # Aggregate-only path
    if args.aggregate_only:
        model_outdirs = {model_name_from_path(p): os.path.join(args.out_root, model_name_from_path(p)) for p in emb_files}
        aggregate_and_plot(args.out_root, model_outdirs, args.relation)
        print("Aggregated existing outputs. Done.")
        return 0

    # SLURM submission path
    if args.submit_slurm:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_root = os.path.join(args.slurm_logs, f"submit_{timestamp}")
        os.makedirs(slurm_root, exist_ok=True)
        job_ids: Dict[str, str] = {}
        for emb_path in emb_files:
            model_name, job_id = submit_single_model_slurm(
                emb_path=emb_path,
                train_dataset=args.train_dataset,
                test_dataset=args.test_dataset,
                relation=args.relation,
                out_root=args.out_root,
                lambda_R=args.lambda_R,
                threshold=args.threshold,
                no_plots=args.no_plots,
                device=args.device,
                num_workers=effective_num_workers,
                slurm_logs=slurm_root,
                partition=args.partition,
                qos=args.qos,
                time=args.time,
                cpus_per_task=args.cpus_per_task,
                gres=args.gres,
                job_extra=args.job_extra,
            )
            job_ids[model_name] = job_id
            print(f"Submitted {model_name}: job {job_id}")

        # Persist job mapping for convenience
        with open(os.path.join(slurm_root, "jobs.json"), "w") as f:
            json.dump(job_ids, f, indent=2)
        print("\nAll jobs submitted. When they finish, run aggregation:")
        agg_cmd = [
            "python3", os.path.join("scripts", "slurm_batch_compare.py"),
            "--emb-dir", args.emb_dir,
            "--train-dataset", args.train_dataset,
            "--test-dataset", args.test_dataset,
            "--relation", args.relation,
            "--out-root", args.out_root,
            "--aggregate-only",
        ]
        print(" ", " ".join(agg_cmd))
        return 0

    # Local execution path (default)
    model_outdirs: Dict[str, str] = {}
    for emb_path in emb_files:
        outdir = run_single_model(
            emb_path=emb_path,
            train_dataset=args.train_dataset,
            test_dataset=args.test_dataset,
            relation=args.relation,
            out_root=args.out_root,
            lambda_R=args.lambda_R,
            threshold=args.threshold,
            no_plots=args.no_plots,
            device=args.device,
            num_workers=effective_num_workers,
        )
        model_outdirs[model_name_from_path(emb_path)] = outdir

    aggregate_and_plot(args.out_root, model_outdirs, args.relation)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
