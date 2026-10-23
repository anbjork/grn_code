import numpy as np
import pandas as pd
import genesnake as gs
from pathlib import Path
import anton_util


reference_networks = {}

anton_util.log_timestamp('Loading reference networks...')
tmp = 'Non-specific-ChIP-seq-network_with_weights'
anton_util.log_timestamp(f'{tmp}...')
true_network_pickle = f'data/{tmp}.pkl'
reference_network_unfiltered = anton_util.unpickle_object(true_network_pickle)
reference_networks[tmp] = reference_network_unfiltered

anton_util.log_timestamp('minaeva...')
reference_networks_path = Path('ground-truth-grns/data/processed/minaeva')
for file in reference_networks_path.iterdir():
    if not file.name.endswith('.gitkeep'):
        refnet = pd.read_csv(file)
        refnet_matrix = gs.util.edgelist_to_matrix(np.array(refnet))
        reference_networks[file.name] = refnet_matrix

anton_util.log_timestamp('cistrome...')
reference_networks_path = Path('ground-truth-grns/data/processed/cistrome')
for file in reference_networks_path.iterdir():
    if not file.name.endswith('.gitkeep'):
        refnet = pd.read_csv(file).drop(['median_regpotential'], axis=1)
        refnet_matrix = gs.util.edgelist_to_matrix(np.array(refnet))
        reference_networks[file.name] = refnet_matrix




# Pre filter the reference networks. Performance optimisation.
# Otherwise it's done repeatedly in benchmarking.
# This relies on there being the same geneset in all datasets
#
# Not verified that this actually increased the speed of the benchmarking.
# Could be that the repeated filtering of the reference networks is not
# the bottleneck
#
# Except for the initial loading. That's faster
#
# I'll keep it in for now. Faster initial loading is still worth it,
# especially when debugging
anton_util.log_timestamp('filtering referece networks...')
path = Path('data/replogle')
data_set_name = 'K562_essential_raw_singlecell_01'
anton_util.log_timestamp(f'Loading {data_set_name}...')
data_sources = anton_util.unpickle_object(
    f'{path}/{data_set_name}_preprocessed_2.pkl')
genes = data_sources[0]['Y'].columns

mock_estimated_network = pd.DataFrame(
    np.zeros((len(genes), len(genes))),
    index=genes,
    columns=genes,
    )

for name, net in reference_networks.items():
    anton_util.log_timestamp(f'{name}...')
    conditions = []
    for axis in range(len(net.shape)):
        conditions.append(np.isin(
            np.array(net.axes[axis]),
            np.array(mock_estimated_network.axes[axis])
            ))
    filtered_net = (
        net.loc[tuple(conditions)]
        )
    reference_networks[name] = filtered_net



anton_util.log_timestamp('Saving networks...')
anton_util.pickle_object(
    reference_networks,
    'data/replogle/compiled_reference_networks.pkl'
    )


