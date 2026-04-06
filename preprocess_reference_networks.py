import numpy as np
import pandas as pd
import genesnake as gs
from pathlib import Path
import anton_util


reference_networks = {}

anton_util.log_timestamp('Loading reference networks...')
tmp = 'Non-specific-ChIP-seq-network_with_weights'
true_network_pickle = f'data/{tmp}.pkl'
reference_network_unfiltered = anton_util.unpickle_object(true_network_pickle)
reference_networks[tmp] = reference_network_unfiltered

reference_networks_path = Path('ground-truth-grns/data/processed/minaeva')
for file in reference_networks_path.iterdir():
    if not file.name.endswith('.gitkeep'):
        refnet = pd.read_csv(file)
        refnet_matrix = gs.util.edgelist_to_matrix(np.array(refnet))
        reference_networks[file.name] = refnet_matrix

reference_networks_path = Path('ground-truth-grns/data/processed/cistrome')
for file in reference_networks_path.iterdir():
    if not file.name.endswith('.gitkeep'):
        refnet = pd.read_csv(file).drop(['median_regpotential'], axis=1)
        refnet_matrix = gs.util.edgelist_to_matrix(np.array(refnet))
        reference_networks[file.name] = refnet_matrix
anton_util.log_timestamp('Reference networks loaded.')



anton_util.pickle_object(
    reference_networks, 
    'data/replogle/compiled_reference_networks.pkl'
    )






