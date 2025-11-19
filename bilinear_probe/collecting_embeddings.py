# collecting_embeddings.py

import argparse
import os
from collections import defaultdict
import json

# Disable MXFP4 integration in Transformers (must be set before importing transformers)
# os.environ.setdefault("HF_USE_MXFP4", "0")
# os.environ.setdefault("HF_MXFP4_AUTO", "0")
# os.environ.setdefault("TRANSFORMERS_DISABLE_MXFP4", "1")

import pandas as pd
import torch
import transformers
from transformers.distributed import DistributedConfig

from tqdm import tqdm

from src import data_utils
from src import inspection_utils
from src import model_utils
from src import tokenization_utils

AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer
try:
    from vllm import LLM  # Optional, only if backend == vllm
except Exception:
    LLM = None

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return int(os.environ.get("RANK", "0")) == 0

def get_parser():
    parser = argparse.ArgumentParser(
        description="Collect entity embeddings from the SOCRATES dataset."
    )
    parser.add_argument("--model_name_or_path", type=str,
                        default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "vllm"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--input_json_paths", type=str,
                        default="counterfact_city-country_train.json,counterfact_city-country_test.json")
    parser.add_argument("--hf_token", type=str,
                        default=os.environ.get("HF_TOKEN", None))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--output_path", type=str,
                        default="entity_embeddings.pt")
    return parser


def collect_unique_entities(data):
    """
    Load a CounterFact-style JSON file and extract pairs of
    (subject, first answer) as strings.

    Returns a list of dicts with keys:
      - "subject": str
      - "answer": str (the first element of "answers")
    """
    entities = set()
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

        entities.add(subject)
        entities.add(answer)
    
    return list(entities)

def llm_correctly_answering(model, tokenizer, is_main, items, batch_size):
    """Return list of booleans for batched correctness checks."""

    def _text_generate_batch(queries):
        if not queries:
            return []
        prompts = []
        for query in queries:
            messages = [
                {"role": "system", "content": "Answer the following questions directly, without any other text before or after your answer."},
                {"role": "user", "content": query}
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        input_device = next(model.parameters()).device
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        tokenizer.padding_side = original_padding_side
        prompt_inputs = {
            k: v.to(input_device) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }
        prompt_length = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=[".", "\n", tokenizer.eos_token],tokenizer=tokenizer,
            )

        new_token_ids = generated_ids[:, prompt_length:].to("cpu")
        return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

    def _is_correct(pred: str, gold_answers):
        pred_l = pred.strip().lower()
        for a in gold_answers:
            idx = pred_l.find(a.lower())
            if idx == 0:
                return True
        return False

    results = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_results = [False] * len(batch_items)
        valid_entries = []  # (relative_idx, item, gold_answers)
        combined_prompts = []
        for rel_idx, item in enumerate(batch_items):
            src_prompt = item.get("src", "")
            rephrase_prompt = item.get("rephrase", "")
            gold_answers = item.get("answers", [])
            has_all_fields = bool(src_prompt and rephrase_prompt and gold_answers)
            if not has_all_fields:
                continue
            combined_prompts.append(src_prompt)
            combined_prompts.append(rephrase_prompt)
            valid_entries.append((rel_idx, item, gold_answers))

        batched_preds = _text_generate_batch(combined_prompts)

        pred_idx = 0
        for rel_idx, item, gold_answers in valid_entries:
            if pred_idx + 1 >= len(batched_preds):
                break
            src_pred = batched_preds[pred_idx]
            rephrase_pred = batched_preds[pred_idx + 1]
            pred_idx += 2
            batch_results[rel_idx] = (
                _is_correct(src_pred, gold_answers)
                and _is_correct(rephrase_pred, gold_answers)
            )

        results.extend(batch_results)

    return results


def extract_entity_embeddings(entities, model, tokenizer, batch_size, is_main):
    """
    Extract embeddings for entities.
    Only main process does the actual work, but all GPUs participate in computation.
    """
    entity_embeddings = {}
    prompts = [f" {entity}" for entity in entities]

    if is_main:
        print(f"Extracting embeddings for {len(entities)} unique entities...")

    iterator = range(0, len(prompts), batch_size)
    if is_main:
        iterator = tqdm(iterator, desc="Processing batches")
    
    for i in iterator:
        batch_prompts = prompts[i:i + batch_size]
        batch_entities = entities[i:i + batch_size]
        
        # All GPUs participate here via tp_plan="auto"
        layer_embeddings = inspection_utils.get_resids(
            prompts=batch_prompts,
            subject_prompts=None,
            model=model,
            tokenizer=tokenizer,
            inner_batch_size=batch_size,
            pos_slice=-1,
            padding_side="left"
        )
        
        # Process each entity in the batch (only main needs results)
        if is_main:
            for j, entity in enumerate(batch_entities):
                entity_embeddings[entity] = {}
                for layer_idx in range(len(layer_embeddings)):
                    emb = layer_embeddings[layer_idx].detach().cpu()
                    entity_embeddings[entity][f'layer_{layer_idx}'] = emb
    
    return entity_embeddings


def load_model_and_tokenizer(args):
    """Load model with tensor parallelism enabled"""
    model_utils.set_random_seed(42)
    
    is_main = is_main_process()
    
    if is_main:
        print(vars(args))

    model_name_or_path = args.model_name_or_path
    model_name = "/".join(model_name_or_path.strip("/").split("/")[-2:])
    safe_model_name = model_name.replace("/", "--")

    if is_main:
        print(f"Loading {model_name_or_path}")
    
    kwargs = {}
    if args.revision:
        kwargs = {"revision": args.revision}
        if is_main:
            print(f"Revision: {args.revision}")
        safe_model_name += f".{args.revision}"

    kwargs["dtype"] = torch.bfloat16
    kwargs["tp_plan"] = "auto"
    if not "openai/gpt-oss" in model_name_or_path:
        kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"
        kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=1)
        

    if args.backend == "hf":
        # Get local rank for device placement
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        
        # Initialize process group for tensor parallelism
        # This is done internally by tp_plan="auto", but we need to set it up
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        
        if is_main:
            print(f"Loading model with tp_plan='auto' across {torch.distributed.get_world_size()} GPUs")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=args.hf_token,
            **kwargs,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=args.hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
    elif args.backend == "vllm":
        if LLM is None:
            raise ImportError("vLLM not installed.")
        if os.environ.get("HF_TOKEN", None) is None and args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token
        
        # vLLM handles its own tensor parallelism
        model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **kwargs,
        )
        tokenizer = model.get_tokenizer()
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    return model, tokenizer


def main(args):
    # Check if we're the main process
    is_main = is_main_process()
    
    # Get distributed info (set by torchrun)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if is_main:
        print(f"Running with {world_size} total processes")
        print(f"Rank: {rank}, Local Rank: {local_rank}")

    # Load model and tokenizer (all ranks do this for tp_plan="auto")
    model, tokenizer = load_model_and_tokenizer(args)
    all_data = []
    split_labels = []
    correct_indices = defaultdict(list)
    entities = []

    for input_json_path in args.input_json_paths.split(","):
        split_name = "train" if "train" in input_json_path.lower() else (
            "test" if "test" in input_json_path.lower() else os.path.splitext(os.path.basename(input_json_path))[0]
        )
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_data.extend(data)
        split_labels.extend([split_name] * len(data))

    corrects = []
    for start in range(0, len(all_data), args.eval_batch_size):
        batch_items = all_data[start:start + args.eval_batch_size]
        batch_results = llm_correctly_answering(
            model,
            tokenizer,
            is_main,
            batch_items,
            args.eval_batch_size,
        )
        corrects.extend(batch_results)
        for offset, is_correct in enumerate(batch_results):
            if is_correct:
                idx = start + offset
                split_name = split_labels[idx] if idx < len(split_labels) else "unknown"
                correct_indices[split_name].append(idx)

    flat_correct_indices = sorted(
        idx for indices in correct_indices.values() for idx in indices
    )
    filtered_all_data = [all_data[idx] for idx in flat_correct_indices]
    entities = collect_unique_entities(filtered_all_data)
        
    if is_main:
        total_correct = len(flat_correct_indices)
        accuracy = total_correct / len(all_data) if all_data else 0.0
        print(f"Accuracy on provided data: {accuracy:.4f} ({total_correct}/{len(all_data)})")
        split_counts = defaultdict(int)
        for label in split_labels:
            split_counts[label] += 1
        
        for split, total in split_counts.items():
            n_correct = len(correct_indices[split])
            acc = n_correct / total if total > 0 else 0.0
            print(f"Accuracy on {split} split: {acc:.4f} ({n_correct}/{total})")
        print(f"Found {len(entities)} unique entities from correctly answered items.")

    if args.backend == "hf" and torch.distributed.is_initialized():
        entities_list = [entities]
        torch.distributed.broadcast_object_list(entities_list, src=0)
        entities = entities_list[0]

    if args.backend == "vllm" and is_main:
        print("vLLM backend: Note that inspection_utils.get_resids may need adaptation.")

    # The loop inside this function is also a collective operation.
    entity_embeddings = extract_entity_embeddings(
        entities, model, tokenizer, args.batch_size, is_main
    )

    # --- Main process saves results ---
    if is_main:
        output_path = args.output_path
        print(f"Saving embeddings to {output_path}")
        
        num_layers = 0
        if entity_embeddings:
            first_entity_key = next(iter(entity_embeddings), None)
            if first_entity_key:
                num_layers = len([k for k in entity_embeddings[first_entity_key].keys() 
                                 if k.startswith('layer_')])
        
        torch.save(
            {
                "meta": {
                    "model": args.model_name_or_path,
                    "num_layers": num_layers,
                    "num_entities": len(entity_embeddings),
                    "correct_indices": {
                        split: indices for split, indices in correct_indices.items()
                    },
                },
                "entities": entity_embeddings,
            },
            output_path,
        )
        print("Done.")
    
    # Ensure all ranks finish before cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(get_parser().parse_args())