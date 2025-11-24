import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import re

def get_layer_num(filename):
    match = re.search(r'layer(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return -1

def extract_metrics(data, prefix=''):
    metrics = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if key.endswith('_acc') or key == 'edit_success':
                if isinstance(value, list):
                    metrics[new_key] = np.mean(value)
                elif isinstance(value, (int, float)):
                    metrics[new_key] = value
            elif isinstance(value, (dict, list)):
                 metrics.update(extract_metrics(value, new_key))
    elif isinstance(data, list):
         # If it's a list of dicts (like the main data), we don't recurse here for metrics extraction
         # This function is intended for the 'post' dictionary of a single case
         pass
    return metrics

def process_results(results_dir):
    # Structure: results/{model_name}/{dataset_name}/lr_{lr}/layer{num}.json
    
    # Load bilinear probe results
    bilinear_probe_path = 'bilinear_probe/outputs/truncated_person-city_wikipedia/best_variance_case_ids.json'
    try:
        with open(bilinear_probe_path, 'r') as f:
            bilinear_data = json.load(f)
    except Exception as e:
        print(f"Error loading bilinear probe data: {e}")
        bilinear_data = {}

    # Find all models
    # Structure: results/{edit_method}/{model_name}/{dataset_name}/lr_{lr}/layer{num}.json
    model_info = []
    if os.path.exists(results_dir):
        for edit_method in os.listdir(results_dir):
            em_path = os.path.join(results_dir, edit_method)
            if os.path.isdir(em_path):
                for model_name in os.listdir(em_path):
                    m_path = os.path.join(em_path, model_name)
                    if os.path.isdir(m_path):
                        model_info.append((model_name, m_path))
    
    # We want to group by dataset_name
    dataset_data = {} # {dataset_name: {model_name: {layer_num: metrics_dict}}}
    best_editing_cases = {} # {dataset_name: {model_name: {'max_success': float, 'case_ids': set}}}

    for model_name, model_path in model_info:
        dataset_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for dataset_name in dataset_dirs:
            if dataset_name not in dataset_data:
                dataset_data[dataset_name] = {}
                best_editing_cases[dataset_name] = {}
            
            if model_name not in dataset_data[dataset_name]:
                dataset_data[dataset_name][model_name] = {}
                best_editing_cases[dataset_name][model_name] = {'max_success': -1.0, 'case_ids': set()}
            
            dataset_path = os.path.join(model_path, dataset_name)
            lr_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('lr_')]
            
            for lr_dir in lr_dirs:
                lr_path = os.path.join(dataset_path, lr_dir)
                layer_files = glob.glob(os.path.join(lr_path, 'layer*.json'))
                
                for layer_file in layer_files:
                    layer_num = get_layer_num(os.path.basename(layer_file))
                    if layer_num == -1:
                        continue
                    
                    try:
                        with open(layer_file, 'r') as f:
                            data = json.load(f)
                        
                        if not data:
                            continue
                        
                        # Calculate average metrics for this file
                        file_metrics = {}
                        count = 0
                        
                        for item in data:
                            if 'post' in item:
                                item_metrics = extract_metrics(item['post'])
                                for k, v in item_metrics.items():
                                    if k not in file_metrics:
                                        file_metrics[k] = []
                                    file_metrics[k].append(v)
                                count += 1
                        
                        if count == 0:
                            continue

                        avg_metrics = {k: np.mean(v) for k, v in file_metrics.items()}
                        avg_success = avg_metrics.get('edit_success', 0.0)
                        
                        # Update max success for this layer (store all metrics)
                        current_best = dataset_data[dataset_name][model_name].get(layer_num, {})
                        current_max = current_best.get('edit_success', -1.0)
                        
                        if avg_success > current_max:
                            dataset_data[dataset_name][model_name][layer_num] = avg_metrics

                        # Update best editing cases for this model/dataset
                        if avg_success > best_editing_cases[dataset_name][model_name]['max_success']:
                            # Extract case_ids where edit_success is 1
                            successful_cases = {item['case_id'] for item in data if 'post' in item and item['post'].get('edit_success') == 1}
                            best_editing_cases[dataset_name][model_name] = {
                                'max_success': avg_success,
                                'case_ids': successful_cases,
                                'layer': layer_num,
                                'lr': lr_dir
                            }
                            
                    except Exception as e:
                        print(f"Error reading {layer_file}: {e}")

    # Compare and Print Table
    print("\nComparison of Bilinear Probe vs Best Editing Case IDs:")
    print(f"{'Dataset':<30} {'Model':<30} {'Bilinear Count':<15} {'Editing Count':<15} {'Overlap':<10} {'Jaccard':<10} {'% Bilinear Covered':<20}")
    print("-" * 130)

    for dataset_name in best_editing_cases:
        for model_name in best_editing_cases[dataset_name]:
            editing_info = best_editing_cases[dataset_name][model_name]
            editing_ids = editing_info['case_ids']
            
            # Match model name with bilinear data keys
            bilinear_key = model_name
            
            bilinear_ids = set()
            if bilinear_key in bilinear_data:
                 bilinear_ids = set(bilinear_data[bilinear_key]['case_ids'])
            else:
                # Try to find a key that contains the model name as a fallback
                for key in bilinear_data:
                    if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                        bilinear_ids = set(bilinear_data[key]['case_ids'])
                        break
            
            if not bilinear_ids:
                print(f"No bilinear data for {model_name} (mapped to {bilinear_key})")
                continue

            overlap = editing_ids.intersection(bilinear_ids)
            overlap_count = len(overlap)
            bilinear_count = len(bilinear_ids)
            editing_count = len(editing_ids)
            
            jaccard = overlap_count / len(editing_ids.union(bilinear_ids)) if len(editing_ids.union(bilinear_ids)) > 0 else 0
            percent_covered = (overlap_count / bilinear_count * 100) if bilinear_count > 0 else 0
            
            print(f"{dataset_name:<30} {model_name:<30} {bilinear_count:<15} {editing_count:<15} {overlap_count:<10} {jaccard:<10.4f} {percent_covered:<20.2f}")


    # Plotting
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all unique metrics across all datasets and models
    all_metrics = set()
    for dataset_name, models_data in dataset_data.items():
        for model_name, layers_data in models_data.items():
            for layer_num, metrics in layers_data.items():
                all_metrics.update(metrics.keys())

    for metric in all_metrics:
        for dataset_name, models_data in dataset_data.items():
            plt.figure(figsize=(10, 6))
            has_data = False
            
            for model_name, layers_data in models_data.items():
                if not layers_data:
                    continue
                    
                sorted_layers = sorted(layers_data.keys())
                # Get metric value, default to 0 if not present (though it should be if consistent)
                metric_values = [layers_data[l].get(metric, 0.0) for l in sorted_layers]
                
                if any(v > 0 for v in metric_values): # Only plot if there's some data
                    plt.plot(sorted_layers, metric_values, marker='o', label=model_name)
                    has_data = True
            
            if has_data:
                plt.title(f"{dataset_name} - {metric}")
                plt.xlabel('Layer')
                plt.ylabel(f'Best Average {metric}')
                plt.legend()
                plt.grid(True)
                
                # Clean metric name for filename
                safe_metric = metric.replace('/', '_').replace(' ', '_')
                save_path = os.path.join(output_dir, f"{dataset_name}_{safe_metric}.png")
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
            
            plt.close()

if __name__ == "__main__":
    process_results('results')
