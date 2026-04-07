import anndata as ad
from pathlib import Path
import anton_util


data_set_name = 'K562_essential_raw_singlecell_01'
anton_util.log_timestamp(f'Loading {data_set_name}...')
# pseudo_bulk_path = Path(f'data/replogle/{data_set_name}_pseudo_bulk.h5ad')
# adata = ad.read_h5ad(pseudo_bulk_path)
preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed.h5ad')
adata = ad.read_h5ad(preprocessed_path)



anton_util.log_timestamp('subsetting...')
n_cells = adata.shape[0] * 0.1
import random
iis = random.sample(range(adata.shape[0]), int(n_cells))
# bool_iis = np.zeros(adata.shape[0], dtype=bool)
# bool_iis[iis] = True
adata = adata[iis, :]


anton_util.log_timestamp('saving...')
checkpoint_path = Path(f'data/replogle/{data_set_name.split(".")[0]}_preprocessed_subset.h5ad')
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(checkpoint_path)




