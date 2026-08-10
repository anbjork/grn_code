set -e

cd ./src/grn_code/data_simulation/
bash run_genespider_simulation.sh
cd -

python preprocess_simulated_data_and_networks.py
python inference_simulated.py
python benchmark_simulated.py
python append_results_with_pipeline.py
python compile_results_simulated.py
python plot_metrics.py


