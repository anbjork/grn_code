
set -e

python preprocess_essential_set_2.py
python inference_essential_set.py
python benchmark_essential_set.py
python compile_results_essential_set.py

