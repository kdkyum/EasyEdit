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

    # Iterate over k_{rank} directories
    k_dirs = glob.glob(os.path.join(root_dir, "k_*"))
    
    if not k_dirs:
        print(f"No k_* directories found in {root_dir}")
        return

    for k_dir in k_dirs:
        dir_name = os.path.basename(k_dir)
        # Extract rank from directory name "k_{rank}"
        try:
            rank = int(dir_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping directory {dir_name}, cannot parse rank.")
            continue
            
        # Iterate over model directories inside k_{rank}
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
                
                max_acc = df["accuracy"].max()
                
                if model_name not in results:
                    results[model_name] = {}
                
                results[model_name][rank] = max_acc
                
            except Exception as e:
                print(f"Error processing {model_dir}: {e}")
                continue

    if not results:
        print(f"No results extracted from {root_dir}")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Sort models for consistent legend
    sorted_models = sorted(results.keys())
    
    for model_name in sorted_models:
        data = results[model_name]
        if not data:
            continue
            
        # Sort by rank
        sorted_ranks = sorted(data.keys())
        accuracies = [data[r] for r in sorted_ranks]
        
        plt.plot(sorted_ranks, accuracies, marker='o', label=model_name)

    plt.xlabel("Rank (k)")
    plt.ylabel("Best Accuracy (Max across layers)")
    plt.title(f"Effect of Rank on Relation Reconstruction Accuracy\n({os.path.basename(root_dir)})")
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2) # Optional: log scale if ranks are 1, 2, 4, 8...
    
    output_plot_path = os.path.join(root_dir, "rank_vs_accuracy_plot.png")
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Rank vs Accuracy from outputs directory containing k_{rank} subfolders.")
    parser.add_argument("root_dirs", nargs='+', help="One or more root directories containing k_{rank} subdirectories")
    args = parser.parse_args()

    for root_dir in args.root_dirs:
        print(f"Processing {root_dir}...")
        process_root_dir(root_dir)

if __name__ == "__main__":
    main()
