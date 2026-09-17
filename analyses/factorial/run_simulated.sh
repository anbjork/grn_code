set -e

python configure_simulations.py

python ../../src/grn_code/data_simulation/simulate_genespider.py
python ../../src/grn_code/data_simulation/gather_simulation_data.py

python pipeline_configuration.py

python ../../src/grn_code/preprocess_simulated_data_and_networks.py
python ../../src/grn_code/inference_simulated.py
python ../../src/grn_code/benchmark_simulated.py

python ../../src/grn_code/append_results_with_pipeline.py
python ../../src/grn_code/compile_results_simulated.py


