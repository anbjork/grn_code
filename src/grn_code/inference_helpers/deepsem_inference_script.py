
# DeepSem is not installable and needs sys.path hacks from hell to run.
# Running it in a subprocess to not corrupt environment of the pipeline.
# This is the helper script for that

import argparse
import sys

input_file_path = sys.argv[1]
output_directory = sys.argv[2]

# Path hack, because DeepSEM repo is not installable
from grn_code.paths_anchor import repo_root
deepsem_root = repo_root / 'DeepSEM'
sys.path.insert(0, str(deepsem_root))

from src.DeepSEM_cell_type_test_non_specific_GRN_model import test_non_celltype_GRN_model

opt = argparse.Namespace(
    task='non_celltype_GRN',
    setting='test',
    n_epochs=120,
    # debug
    # n_epochs=20,
    batch_size=64,
    alpha=100,
    beta=1,
    lr=1e-4,
    lr_step_size=0.99,
    gamma=0.95,
    n_hidden=128,
    K=1,
    K1=1,
    K2=2,
    net_file=None,
)
opt.data_file = input_file_path
opt.save_name = output_directory

model = test_non_celltype_GRN_model(opt)
model.train_model()


