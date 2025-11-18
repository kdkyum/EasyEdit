#!/usr/bin/env python3
"""
Download one or more Hugging Face model repositories without loading them into memory.

Two modes are supported:
- snapshot (default): uses huggingface_hub.snapshot_download to fetch files to disk only
- transformers-meta: uses transformers.from_pretrained with device_map="meta" and offload_state_dict=True

Examples:
  # default: snapshot download of multiple repos
  python scripts/download_models.py \
    -m allenai/OLMo-2-1124-13B \
    -m allenai/OLMo-2-1124-13B-Instruct \
    -m mistralai/Mistral-Large-Instruct-2411 \
    -m openai/gpt-oss-20b \
    -m openai/gpt-oss-120b \
    --out ./hf_models

  # use transformers-only path (still does not materialize weights in memory)
  python scripts/download_models.py -m sshleifer/tiny-gpt2 --mode transformers-meta --out ./hf_models

Notes:
- For gated models, ensure you have accepted the license on the model page and are logged in (HF_TOKEN env var or `huggingface-cli login`).
- Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads (requires `pip install hf-transfer`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_HF_DATASETS_CACHE = "/ptmp/kdkyum/huggingface_cache/datasets"
DEFAULT_HF_HOME = "/ptmp/kdkyum/huggingface_cache/hf_home"


def ensure_default_hf_env():
    """Set default HF cache directories if not already provided by the environment.

    - HF_DATASETS_CACHE: where the datasets library caches data
    - HF_HOME: base dir for Hugging Face config and hub cache (under $HF_HOME/hub)
    """
    # Respect user overrides when already set in the environment
    os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)

    try:
        Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model repo ID on Hugging Face (e.g., allenai/OLMo-2-1124-13B). Repeat for multiple.",
    )
    p.add_argument(
        "--mode",
        choices=["snapshot", "transformers-meta"],
        default="snapshot",
        help="Download method. 'snapshot' = hub-only, 'transformers-meta' = transformers with device_map=meta.",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default=str(Path.cwd() / "hf_models"),
        help="Base output directory to place model files (one subdir per repo).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (branch/tag/commit) to pin.",
    )
    p.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN env var if set.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="When using transformers-meta mode, allow custom modeling code.",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional globs to include (snapshot mode). Example: --include '*.safetensors' config.json tokenizer.json tokenizer.model tokenizer_config.json "
            "If omitted, the entire repo is downloaded."
        ),
    )
    return p.parse_args()


def repo_to_local_path(base: Path, repo_id: str) -> Path:
    # Create a deterministic local path per repo id
    owner, name = repo_id.split("/", 1)
    return base / owner / name


def download_snapshot(repo_id: str, out_dir: Path, revision: Optional[str], token: Optional[str], include: Optional[List[str]]):
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("ERROR: huggingface_hub is required for snapshot mode. Install via `pip install huggingface_hub`.", file=sys.stderr)
        raise

    target_dir = repo_to_local_path(out_dir, repo_id)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[snapshot] Downloading {repo_id} -> {target_dir}")
    # local_dir_use_symlinks=False ensures real files are placed under target_dir
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        allow_patterns=include,
    )
    print(f"[snapshot] Done: {local_path}")


def download_transformers_meta(repo_id: str, out_dir: Path, revision: Optional[str], token: Optional[str], trust_remote_code: bool):
    # This path uses transformers to trigger downloads but keeps weights off memory using device_map="meta".
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    target_dir = repo_to_local_path(out_dir, repo_id)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[transformers-meta] Preparing {repo_id} -> {target_dir}")

    # Download config and tokenizer artifacts
    _ = AutoConfig.from_pretrained(repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code)
    _ = AutoTokenizer.from_pretrained(repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code)

    # Trigger weight file downloads without loading tensors into RAM
    # offload_state_dict=True ensures we don't materialize params
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
        device_map="meta",
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )
    # Immediately drop any handles
    del model
    print(f"[transformers-meta] Done: {target_dir}")


def main():
    # Ensure default caches per user preference unless already exported
    ensure_default_hf_env()

    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for repo_id in args.models:
        try:
            if args.mode == "snapshot":
                download_snapshot(repo_id, out_dir, args.revision, args.token, args.include)
            else:
                download_transformers_meta(repo_id, out_dir, args.revision, args.token, args.trust_remote_code)
        except Exception as e:
            print(f"ERROR downloading {repo_id}: {e}", file=sys.stderr)
            failures.append((repo_id, str(e)))

    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for repo_id, err in failures:
            print(f"- {repo_id}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
