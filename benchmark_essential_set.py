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

true_network_pickle = 'data/Non-specific-ChIP-seq-network_with_weights.pkl'
reference_network_unfiltered = anton_util.unpickle_object(true_network_pickle)

stats = {}
for method, estimated_network in estimated_networks.items():

	if filter_reference:
		conditions = []
		for axis in range(len(reference_network_unfiltered.shape)):
			conditions.append(np.isin(
				np.array(reference_network_unfiltered.axes[axis]),
				np.array(estimated_network.axes[axis])
				))
		reference_network = (
			reference_network_unfiltered.loc[tuple(conditions)]
			)
	else:
		reference_network = reference_network_unfiltered

	tmp = gs.util.harmonise_networks((
		estimated_network,
		reference_network))
	harmonised_estimated_network, harmonised_reference_network = tmp

	tmp = gs.benchmarking.benchmark(
		estimated_network=harmonised_estimated_network,
		reference_network=harmonised_reference_network.astype(bool),
		plot_dir=benchmark_output_dir / 'individual_perturbation_methods',
		method_name=method
		)
	stats[method] = {
		'dataset': 'replogle_essential',
		'method': method,
		**tmp
		}

df = pd.DataFrame(stats).T
df.index.name = r'method \ metric'
df.to_csv(benchmark_output_dir / 'perturbation_methods_stats.csv')
anton_util.pickle_object(df, benchmark_output_dir / 'perturbation_methods_stats.pkl')
