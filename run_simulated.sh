set -e

# python preprocess_simulated_data_and_networks.py
python inference_simulated.py
python benchmark_simulated.py
python compile_results_simulated.py
python plot_metrics.py


