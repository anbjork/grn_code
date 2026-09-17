

# preprocessing_options = {
#         'cell normalised': [False, True],
#         # 'read normalised': [False, True],
#         'read normalised': [False],
#         'transform 1': ['none', 'log1p'],
#         'transform 2': ['none', 'zscores'],
#         # 'pseudo_bulk': [False, 1, 2, 3, 5, 10],
#         'pseudo_bulk': [False, 5],
#         'shuffle': [False],
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


inference_functions = [
        'fast_methods_inference',
        'random_inference',
        'correlation_inference',
        'perfect_inference',
        # 'zscore_max_variants',
        # 'zscore_without_controls',
        # 'lsco_T_without_controls',
        # 'inspre_inference',
        # 'inspre_inference_hdf5',
        # 'psgrn_inference',
        # 'genie3_inference',
        # 'deepsem_inference',
        # 'dspin_inference',
        # 'dspin_inference_wrapper',
        ]


config = {
        'preprocessing_options': preprocessing_options,
        'inference_functions': inference_functions,
        }
import anton_util
from grn_code.pipeline_code import pipeline_base_path
pipeline_base_path.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(config, pipeline_base_path / 'pipeline_configuration.pkl')




