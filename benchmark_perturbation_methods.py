
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genesnake as gs
import pickle
from pathlib import Path
from copy import deepcopy
import anton_util



filter_reference = True




benchmark_output_dir = Path('outputs')
[d.mkdir(exist_ok = True) for d in [benchmark_output_dir]]

### Load networks

estimated_networks = anton_util.unpickle_object(
    benchmark_output_dir / 'estimated_networks.pkl')


# # Dspin takes 3 days to run, so using the results from before.
# # I copied them over from version 13 of Replogle data analysis,
# # see separate script for details on that.
# dspin_network_dir = benchmark_output_dir
# cur_j = anton_util.unpickle_object(dspin_network_dir / 'cur_j_500.pkl')
# # model = anton_util.unpickle_object(dspin_network_dir / 'model_500.pkl')
# genes = anton_util.unpickle_object(dspin_network_dir / 'gene_names_500.pkl')
# dspin_network = pd.DataFrame(
#   index = genes,
#   columns = genes,
#   data = cur_j,
#   )
# estimated_networks['dspin'] = dspin_network

# Add randomly reweighted networks
# m = 'dspin'
# net = estimated_networks[m]
# for ii in range(3):
#   estimated_networks[f'{m}_random_{str(ii)}'] = pd.DataFrame(
#       data = np.random.random_sample(size = net.shape),
#       index = deepcopy(net.index),
#       columns = deepcopy(net.columns)
#       )

true_network_pickle = 'data/Non-specific-ChIP-seq-network_with_weights.pkl'
reference_network_unfiltered = anton_util.unpickle_object(true_network_pickle)

stats = {}
for method, estimated_network in estimated_networks.items():

    # anton_util.log_timestamp('prepping')

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

    tmp = gs.util.conversions.harmonise_networks((
        estimated_network, 
        reference_network))
    harmonised_estimated_network, harmonised_reference_network = tmp


    # import ipdb; ipdb.set_trace()


    # from IPython import embed; embed(colors = 'Linux')

    # anton_util.log_timestamp('benching')

    tmp = gs.benchmarking.benchmark(
        estimated_network = harmonised_estimated_network,
        reference_network = harmonised_reference_network.astype(bool),
        plot_dir = benchmark_output_dir / 'individual_perturbation_methods',
        method_name = method
        )
    stats[method] = {
        'dataset': 'replogle',
        'method': method,
        **tmp
        }

    # anton_util.log_timestamp(f'{method} finished')


df = pd.DataFrame(stats).T
df.index.name = r'method \ metric'
df.to_csv(benchmark_output_dir / 'perturbation_methods_stats.csv')
anton_util.pickle_object(df, benchmark_output_dir / 'perturbation_methods_stats.pkl')


