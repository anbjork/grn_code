import numpy as np
import anndata as ad
from pathlib import Path

data_set_name = 'K562_essential_raw_singlecell_01.h5ad'
checkpoint_path = Path(f'data/replogle/{data_set_name.split('.')[0]}_preprocessed.h5ad')
adata = ad.read_h5ad(checkpoint_path)
from collections import Counter
aa = Counter(Counter(adata.obs.gene).values())
counts, count_counts = zip(*aa.items())
order = sorted(range(len(counts)), key = lambda ii: counts[ii])
sorted_counts = np.array(counts)[order]
sorted_count_counts = np.array(count_counts)[order]
sorted_both = list(zip(sorted_counts.tolist(), sorted_count_counts.tolist()))



