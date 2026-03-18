
# This is run with the replogle grn venv, since it has a compatible
# version of dspin and its dependencies installed. 
# I think an old version of AnnData is the culprit in this case, 
# based on the error message using a more recent Python environment.
# Needed to extract the genes from the dspin model object.

import numpy as np
from pathlib import Path
import anton_util

outputs = Path('outputs')

# Dspin takes 3 days to run, so using the results from before.
# I copied them over from version 13 of Replogle data analysis,
# see separate script for details on that.
dspin_network_dir = outputs
model = anton_util.unpickle_object(dspin_network_dir / 'model_500.pkl')
genes = np.array(model.adata.var.gene_name)
anton_util.pickle_object(genes, dspin_network_dir / 'gene_names_500.pkl')


