import os.path
import numpy as np

import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import argparse
from pathlib import Path

def _mean_or_none(values):
    return float(np.mean(values)) if values else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_path', default='./output/metrics.json', type=str)
    parser.add_argument('--chat_mode', action='store_true')

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    else:
        raise NotImplementedError

    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data_json = json.load(f)

    # filtering only correct instances by the LM we want to edit
    correct_indices = np.load(args.correct_indices_path)["test_indices"]
    test_data = [test_data_json[i] for i in correct_indices]

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = [test_data_['src'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
    target_new = [edit_data_['alt'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'two_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    subject = [edit_data_['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    train_ds = None

    if hasattr(hparams, 'lrs') and hparams.lrs:
        lrs = hparams.lrs
    else:
        lrs = [hparams.lr] if hasattr(hparams, 'lr') else [None]

    original_metrics_save_path = Path(args.metrics_save_path)

    for lr in lrs:
        if lr is not None:
            hparams.lr = lr
            metrics_path = original_metrics_save_path.parent / f"lr_{lr}" / original_metrics_save_path.name
        else:
            metrics_path = original_metrics_save_path

        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new, 
            subject=subject,
            train_ds=train_ds,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            chat_mode=args.chat_mode
        )

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
