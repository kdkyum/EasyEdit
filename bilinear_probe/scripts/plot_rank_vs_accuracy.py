import argparse
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import re

def process_root_dir(root_dir):
    # Data structure: results[model_name][rank] = max_accuracy
    results = {}
    mse_results = {}  # model_name -> rank -> min_test_mse
    auc_results = {}  # model_name -> rank -> max_auc_pr
    best_configs = {}
    layer_results = {}

    # Detect mode and collect directories
    variance_dirs = glob.glob(os.path.join(root_dir, "variance_*"))
    k_dirs = glob.glob(os.path.join(root_dir, "k_*"))
    
    if k_dirs:
        mode = "k"
        target_dirs = k_dirs
        xlabel = "Rank k"
        param_key = "k"
    elif variance_dirs:
        mode = "variance"
        target_dirs = variance_dirs
        xlabel = "Variance Threshold"
        param_key = "variance_threshold"
    else:
        print(f"No variance_* or k_* directories found in {root_dir}")
        return

    for k_dir in target_dirs:
        dir_name = os.path.basename(k_dir)
        # Extract value from directory name
        try:
            if mode == "k":
                val = int(dir_name.split("_")[1])
            else:
                val = float(dir_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping directory {dir_name}, cannot parse value.")
            continue
            
        # Iterate over model directories inside
        model_dirs = glob.glob(os.path.join(k_dir, "*_embeddings"))
        
        for model_dir in model_dirs:
            model_folder_name = os.path.basename(model_dir)
            # Clean model name
            model_name = model_folder_name.replace("_embeddings", "")
            
            summary_path = os.path.join(model_dir, "summary.json")
            if not os.path.exists(summary_path):
                # print(f"Warning: summary.json not found in {model_dir}")
                continue
                
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                
                relation = summary.get("relation", "person-city")
                # Handle potential relation name variations in path (e.g. / replaced by _)
                relation_path_name = relation.replace("/", "_")
                
                csv_path = os.path.join(model_dir, relation_path_name, f"metrics_{relation_path_name}.csv")
                
                if not os.path.exists(csv_path):
                    # print(f"Warning: Metrics CSV not found at {csv_path}")
                    continue
                
                df = pd.read_csv(csv_path)
                if "accuracy" not in df.columns:
                    print(f"Warning: 'accuracy' column missing in {csv_path}")
                    continue
                
                best_idx = df["accuracy"].idxmax()
                max_acc = df.loc[best_idx, "accuracy"]
                
                if "test_mse" in df.columns:
                    min_mse = df["test_mse"].min()
                    if model_name not in mse_results:
                        mse_results[model_name] = {}
                    mse_results[model_name][val] = min_mse

                if "auc_pr" in df.columns:
                    max_auc = df["auc_pr"].max()
                    if model_name not in auc_results:
                        auc_results[model_name] = {}
                    auc_results[model_name][val] = max_auc

                if "layer" in df.columns:
                    best_layer = df.loc[best_idx, "layer"]
                else:
                    best_layer = best_idx

                if model_name not in results:
                    results[model_name] = {}
                
                results[model_name][val] = max_acc

                if model_name not in layer_results:
                    layer_results[model_name] = {}
                
                for idx, row in df.iterrows():
                    current_layer = int(row["layer"]) if "layer" in df.columns else idx
                    current_acc = row["accuracy"]
                    if current_layer not in layer_results[model_name] or current_acc > layer_results[model_name][current_layer]:
                        layer_results[model_name][current_layer] = current_acc

                if model_name not in best_configs or max_acc > best_configs[model_name]['acc']:
                     json_path = os.path.join(model_dir, relation_path_name, "bilinear_correct_ids.json")
                     best_configs[model_name] = {
                         'acc': max_acc,
                         'param_value': val,
                         'param_type': mode,
                         'layer': int(best_layer),
                         'json_path': json_path
                     }
                
            except Exception as e:
                print(f"Error processing {model_dir}: {e}")
                continue

    if not results:
        print(f"No results extracted from {root_dir}")
        return

    # Plotting Accuracy
    plt.figure(figsize=(10, 6))
    
    # Sort models for consistent legend
    sorted_models = sorted(results.keys())
    
    for model_name in sorted_models:
        data = results[model_name]
        if not data:
            continue
            
        # Sort by value
        sorted_vals = sorted(data.keys())
        accuracies = [data[v] for v in sorted_vals]
        
        # Convert vals to strings and plot with uniform spacing if variance, or linear if k?
        # Keeping uniform spacing logic for now as it handles non-linear steps well
        x_positions = list(range(len(sorted_vals)))
        x_labels = [str(v) for v in sorted_vals]
        plt.plot(x_positions, accuracies, marker='o', label=model_name)
        plt.xticks(x_positions, x_labels, rotation=45 if len(x_labels) > 10 else 0)
        
    plt.xlabel(xlabel)
    plt.ylabel("Best Accuracy (Max across layers)")
    plt.title(f"Effect of {xlabel} on Relation Reconstruction Accuracy\n({os.path.basename(root_dir)})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_plot_path = os.path.join(root_dir, f"{mode}_vs_accuracy_plot.png")
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
    plt.close()

    # Plotting Test MSE
    if mse_results:
        plt.figure(figsize=(10, 6))
        for model_name in sorted_models:
            if model_name not in mse_results:
                continue
            data = mse_results[model_name]
            if not data:
                continue
            
            sorted_vals = sorted(data.keys())
            mses = [data[v] for v in sorted_vals]
            
            x_positions = list(range(len(sorted_vals)))
            x_labels = [str(v) for v in sorted_vals]
            plt.plot(x_positions, mses, marker='o', label=model_name)
            plt.xticks(x_positions, x_labels, rotation=45 if len(x_labels) > 10 else 0)
            
        plt.xlabel(xlabel)
        plt.ylabel("Min Test MSE (Min across layers)")
        plt.title(f"Effect of {xlabel} on Relation Reconstruction Test MSE\n({os.path.basename(root_dir)})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_mse_plot_path = os.path.join(root_dir, f"{mode}_vs_test_mse_plot.png")
        plt.savefig(output_mse_plot_path)
        print(f"Test MSE plot saved to {output_mse_plot_path}")
        plt.close()

    # Plotting AUC-PR
    if auc_results:
        plt.figure(figsize=(10, 6))
        for model_name in sorted_models:
            if model_name not in auc_results:
                continue
            data = auc_results[model_name]
            if not data:
                continue
            
            sorted_vals = sorted(data.keys())
            aucs = [data[v] for v in sorted_vals]
            
            x_positions = list(range(len(sorted_vals)))
            x_labels = [str(v) for v in sorted_vals]
            plt.plot(x_positions, aucs, marker='o', label=model_name)
            plt.xticks(x_positions, x_labels, rotation=45 if len(x_labels) > 10 else 0)
            
        plt.xlabel(xlabel)
        plt.ylabel("Best AUC-PR (Max across layers)")
        plt.title(f"Effect of {xlabel} on Relation Reconstruction AUC-PR\n({os.path.basename(root_dir)})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_auc_plot_path = os.path.join(root_dir, f"{mode}_vs_auc_pr_plot.png")
        plt.savefig(output_auc_plot_path)
        print(f"AUC-PR plot saved to {output_auc_plot_path}")
        plt.close()

    # Plotting Layer vs Best Accuracy (across variances/k)
    plt.figure(figsize=(10, 6))
    
    for model_name in sorted_models:
        if model_name not in layer_results:
            continue
        data = layer_results[model_name]
        if not data:
            continue
            
        sorted_layers = sorted(data.keys())
        accuracies = [data[l] for l in sorted_layers]
        
        plt.plot(sorted_layers, accuracies, marker='o', label=model_name)
        
    plt.xlabel("Layer")
    plt.ylabel(f"Best Accuracy (Max across {mode})")
    plt.title(f"Best Accuracy per Layer across all {mode}s\n({os.path.basename(root_dir)})")
    plt.legend()
    plt.grid(True)
    
    output_layer_plot_path = os.path.join(root_dir, "layer_vs_best_accuracy_plot.png")
    plt.savefig(output_layer_plot_path)
    print(f"Layer plot saved to {output_layer_plot_path}")
    plt.close()

    # Save best case IDs
    output_data = {}
    for model_name, data in best_configs.items():
        if os.path.exists(data['json_path']):
            try:
                with open(data['json_path'], 'r') as f:
                    correct_ids_data = json.load(f)
                
                target_case_ids = []
                for entry in correct_ids_data:
                    if entry['layer'] == data['layer']:
                        target_case_ids = entry['case_ids']
                        break
                
                output_data[model_name] = {
                    'best_param_value': data['param_value'],
                    'param_type': data['param_type'],
                    'best_layer': data['layer'],
                    'max_accuracy': data['acc'],
                    'case_ids': target_case_ids
                }
            except Exception as e:
                print(f"Error reading {data['json_path']}: {e}")
        else:
            print(f"Warning: {data['json_path']} not found for best model result.")
    
    if output_data:
        output_json_path = os.path.join(root_dir, "best_variance_case_ids.json")
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Best case IDs saved to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Rank vs Accuracy from outputs directory containing k_{rank} subfolders.")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing k_{rank} subdirectories")
    args = parser.parse_args()

    for root_dir in args.root_dirs:
        print(f"Processing {root_dir}...")
        process_root_dir(root_dir)

if __name__ == "__main__":
    main()
