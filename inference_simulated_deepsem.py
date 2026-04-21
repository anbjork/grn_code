

from pathlib import Path
import anton_util
import functions
import copy

path = Path('data/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

# # Debug
# data_sources = data_sources[:1]

estimated_networks = []
for ii, data_source in enumerate(data_sources):
    anton_util.log_timestamp(f'dataset {ii}...')
    en = functions.deepsem_inference(data=data_source)

    meta = copy.deepcopy(data_source['meta'])
    meta['method'] = 'deepsem'
    estimated_networks.append({
        'meta': meta,
        'estimated_network': en,
    })

output_dir = Path('inferences/simulated')
output_dir.mkdir(exist_ok=True, parents=True)
p = output_dir / f'estimated_networks.pkl'
if p.exists():
    oldnets = anton_util.unpickle_object(p)
    estimated_networks = oldnets + estimated_networks
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')
anton_util.log_timestamp('Estimated networks saved.')











# Code I got from collaborators

# import os
# import subprocess
# import argparse
# import numpy as np
# import pandas as pd
# import shutil
# from genesnake.util import harmonise_networks
# from genesnake.benchmarking.benchmark import benchmark as gs_benchmark

# def convert_deepsem_tsv_to_csv(tsv_path, csv_path):
#     """Converts DeepSEM TSV output to GeneSNAKE-compatible CSV."""
#     try:
#         df = pd.read_csv(tsv_path, sep='\t')
#         df = df.iloc[:, :3] 
#         df.columns = ['Regulator', 'Target', 'Weight']
#         df['Sign'] = np.sign(df['Weight']).astype(int)
#         df.to_csv(csv_path, index=False)
#         return True
#     except Exception as e:
#         print(f"  -> Error converting DeepSEM output: {e}")
#         return False

# def run_deepsem_and_convert(dataset_name, expr_path, deepsem_dir):
#     """Runs DeepSEM and converts the output internally."""
#     deepsem_main = os.path.join(deepsem_dir, "main.py")


#     # This seems to be the relevant bit
#     # Everything else is pipelining code, path management, file reading, etc
#     # And benchmarking, which I do with another separate script
#     deepsem_cmd = [
#         "python3", deepsem_main,
#         "--task", "non_celltype_GRN",
#         "--data_file", expr_path,
#         "--setting", "default",
#         "--save_name", dataset_name
#     ]
    
#     print(f"  -> Running DeepSEM...")
#     subprocess.run(deepsem_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
    
#     tsv_out = os.path.join(dataset_name, "GRN_inference_result.tsv")
#     if not os.path.exists(tsv_out):
#         print(f"  -> Error: DeepSEM failed to produce TSV for {dataset_name}.")
#         return None

#     print(f"  -> Converting format...")
#     final_csv = os.path.join(dataset_name, "GRN_inference_result.csv")
    
#     if not convert_deepsem_tsv_to_csv(tsv_out, final_csv):
#         return None

#     return final_csv

# def main():
#     parser = argparse.ArgumentParser(description="Streamlined DeepSEM + GeneSNAKE Pipeline")
#     parser.add_argument("--datasets", type=str, default="datasets", help="Path to datasets folder")
#     parser.add_argument("--deepsem", type=str, default="./DeepSEM", help="Path to DeepSEM folder")
#     parser.add_argument("--out", type=str, default="deepsem_benchmark_summary.csv", help="Output summary file")
#     args = parser.parse_args()

#     results = []
#     base_dir = args.datasets

#     for dataset_name in os.listdir(base_dir):
#         ds_path = os.path.join(base_dir, dataset_name)
#         if not os.path.isdir(ds_path):
#             continue
            
#         print(f"\n--- Processing: {dataset_name} ---")

#         expr_file = os.path.join(ds_path, "log_fold_change.csv")
#         gt_file = os.path.join(ds_path, "ground_truth.csv")

#         # Simplified Pre-flight check
#         if not (os.path.exists(expr_file) and os.path.exists(gt_file)):
#             print(f"  -> Missing expression.csv or ground_truth.csv. Skipping.")
#             continue

#         # Step 1: Run Inference & Convert
#         converted_csv = run_deepsem_and_convert(dataset_name, expr_file, args.deepsem)
#         if not converted_csv:
#             continue

#         # Step 2: Benchmark
#         print(f"  -> Benchmarking...")
#         try:
#             gt_df = pd.read_csv(gt_file, index_col=0)
            
#             inf_raw = pd.read_csv(converted_csv)
#             inf_raw.columns = inf_raw.columns.str.strip()
#             inf_df = inf_raw.pivot_table(index='Regulator', columns='Target', values='Weight', aggfunc='max', fill_value=0.0)

#             est, ref = harmonise_networks((inf_df, gt_df))
#             stats = gs_benchmark(estimated_network=est, reference_network=ref.astype(bool), plot_dir=None, method_name='DeepSEM')
            
#             aupr = float(stats.get('AUPR', 0.0))
#             erma = float(stats.get('ERMA', 0.0))
            
#             # Base row dictionary
#             row_data = {
#                 'Directory': dataset_name,
#                 'aupr_vs_random': (aupr / erma) if erma > 0.0 else 0.0
#             }
            
#             # Add all metrics dynamically
#             for k, v in stats.items():
#                 if k != 'f1_scores':
#                     row_data[k] = v

#             results.append(row_data)
#             print(f"  -> Done! AUROC: {stats.get('AUROC', 0):.4f}")
            
#         except Exception as e:
#             print(f"  -> Benchmarking failed: {e}")
#             continue
            
#         finally:
#             # Cleanup
#             if os.path.exists(dataset_name):
#                 shutil.rmtree(dataset_name)

#     # Final Output Formatting
#     if not results:
#         print("\nNo datasets were successfully processed.")
#         return

#     print(f"\nSaving final summary to {args.out}...")
#     df = pd.DataFrame(results)
    
#     # Organize columns logically, prioritizing core metrics
#     core_cols = ['Directory', 'AUROC', 'AUPR', 'aupr_vs_random', 'EPR', 'ERMA', 'F1']
#     existing_core = [c for c in core_cols if c in df.columns]
#     other_cols = [c for c in df.columns if c not in core_cols]
    
#     df = df[existing_core + other_cols]
    
#     # Sort alphabetically by directory name
#     df.sort_values(by=['Directory'], inplace=True)
        
#     df.to_csv(args.out, index=False)
#     print("Pipeline complete.")

# if __name__ == "__main__":
#     main()
