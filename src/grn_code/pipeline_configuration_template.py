
from pathlib import Path

# Path handling
# Configure
output_base_path = Path('outputs')
# Change when knowingly changing structure
pipeline_base_path = Path(f'{output_base_path}/in_pipeline')



# preprocessing_options = {
#         'cell normalised': [False, True],
#         'read normalised': [False, True],
#         'transform 1': ['none', 'log1p'],
#         'transform 2': ['none', 'zscores'],
#         'pseudo_bulk': [False, 1, 2, 3, 5, 10],
#         'shuffle': [False, True],
#         'compute differences': [False, True],
#         }
preprocessing_options = {
        'cell normalised': [True],
        'read normalised': [False],
        'transform 1': ['log1p'],
        'transform 2': ['zscores'],
        'pseudo_bulk': [False],
        'shuffle': [False],
        'compute differences': [False],
        }




from grn_code import functions
inference_functions = [
    functions.fast_methods_inference,
    functions.random_inference,
    functions.correlation_inference,
    functions.perfect_inference,
    functions.zscore_max_variants,
    functions.zscore_without_controls,
    functions.lsco_T_without_controls,
    functions.inspre_inference,
    functions.inspre_inference_hdf5,
    functions.psgrn_inference,
    functions.genie3_inference,
    functions.deepsem_inference,
    functions.dspin_inference,
    functions.dspin_inference_wrapper,
    ]








