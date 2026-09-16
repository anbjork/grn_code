set -e

python configure_simulations_factorial.py
# python configure_simulations_data_cases.py
python simulate_genespider.py
python gather_simulation_data.py

python preprocess_simulated_data_and_networks.py
python inference_simulated.py
python benchmark_simulated.py
python append_results_with_pipeline.py
python compile_results_simulated.py
python plot_metrics.py


