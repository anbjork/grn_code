import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genesnake as gs
import pickle
from pathlib import Path
from copy import deepcopy
import anton_util


filter_reference = True


inference_dir = Path('inferences')
benchmark_output_dir = Path('benchmarks')
[d.mkdir(exist_ok=True) for d in [benchmark_output_dir]]

### Load networks

estimated_networks = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')





reference_networks = {}

true_network_pickle = 'data/Non-specific-ChIP-seq-network_with_weights.pkl'
reference_network_unfiltered = anton_util.unpickle_object(true_network_pickle)
reference_networks['Non-specific-ChIP-seq-network_with_weights'] = reference_network_unfiltered

reference_networks_path = Path('getting_networks/ground-truth-grns/data/processed/minaeva')
for file in reference_networks_path.iterdir():
    if not file.name.endswith('.gitkeep'):
        refnet = pd.read_csv(file)
        refnet_matrix = gs.util.edgelist_to_matrix(np.array(refnet))
        reference_networks[file.name] = refnet_matrix

cistrome_network = Path('getting_networks/ground-truth-grns/data/processed/cistrome/k562_cistrome_regpotential_geq_0.61.csv')
df = pd.read_csv(cistrome_network).drop(['median_regpotential'], axis=1)
reference_networks[cistrome_network.name] = gs.util.edgelist_to_matrix(np.array(df))





stats = {}
for method, estimated_network in estimated_networks.items():
    for ref_name, reference_network_unfiltered_current in reference_networks.items():

        anton_util.log_timestamp(f'Benchmarking {method} against {ref_name}...')

        if filter_reference:
            conditions = []
            for axis in range(len(reference_network_unfiltered_current.shape)):
                conditions.append(np.isin(
                    np.array(reference_network_unfiltered_current.axes[axis]),
                    np.array(estimated_network.axes[axis])
                    ))
            reference_network = (
                reference_network_unfiltered_current.loc[tuple(conditions)]
                )
        else:
            reference_network = reference_network_unfiltered_current

        tmp = gs.util.harmonise_networks((
            estimated_network,
            reference_network))
        harmonised_estimated_network, harmonised_reference_network = tmp

        id = f'{method}_{ref_name}'
        tmp = gs.benchmarking.benchmark(
            estimated_network=harmonised_estimated_network,
            reference_network=harmonised_reference_network.astype(bool),
            plot_dir=benchmark_output_dir / method,
            method_name=id,
            )
        stats[id] = {
            'dataset': 'replogle_essential',
            'method': method,
            'reference_network': ref_name,
            **tmp
            }
        anton_util.log_timestamp(f'Finished benchmarking {method} against {ref_name}')

df = pd.DataFrame(stats).T
df.index.name = r'method \ metric'
df.to_csv(benchmark_output_dir / 'perturbation_methods_stats.csv')
anton_util.pickle_object(df, benchmark_output_dir / 'perturbation_methods_stats.pkl')



