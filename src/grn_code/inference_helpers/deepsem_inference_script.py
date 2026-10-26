
# DeepSem is not installable and needs sys.path hacks from hell to run.
# Running it in a subprocess to not corrupt environment of the pipeline.
# This is the helper script for that

import argparse
import sys

input_file_path = sys.argv[1]
output_directory = sys.argv[2]

# # Save the original working directory so we can restore it later
# original_dir = os.getcwd()
# temp_file = None
# temp_dir = None

# Path hack, because DeepSEM repo is not installable
from grn_code.paths_anchor import repo_root
deepsem_root = repo_root / 'DeepSEM'
sys.path.insert(0, str(deepsem_root))

# Import DeepSEM test module (for inference without ground truth)
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


# # Create temporary file for data
# with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#     temp_file = f.name
#     expression_data.to_csv(f.name)
#     opt.data_file = f.name

# # Create temporary directory for save_name
# temp_dir = tempfile.mkdtemp()
# opt.save_name = temp_dir

model = test_non_celltype_GRN_model(opt)
model.train_model()


# result_file = os.path.join(temp_dir, "GRN_inference_result.tsv")












    #
    #
    # # Create temporary file for data
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    #     temp_file = f.name
    #     expression_data.to_csv(f.name)
    #     opt.data_file = f.name
    #
    #
    # # Create temporary directory for save_name
    # temp_dir = tempfile.mkdtemp()
    # opt.save_name = temp_dir
    #
    # import subprocess
    # subprocess.run([
    #
    #
    #
    # # Read the output - based on collaborator code, it creates "GRN_inference_result.tsv"
    # result_file = os.path.join(temp_dir, "GRN_inference_result.tsv")
    # if os.path.exists(result_file):
    #     result_df = pd.read_csv(result_file, sep='\t')
    # else:
    #     raise RuntimeError("DeepSEM did not produce expected output file")
    #
    # # # finally:
    # # # Restore the original working directory
    # # os.chdir(original_dir)
    # # # Cleanup temp file and temp dir
    # # if temp_file is not None and os.path.exists(temp_file):
    # #     os.unlink(temp_file)
    # # if temp_dir is not None and os.path.exists(temp_dir):
    # #     shutil.rmtree(temp_dir)
    #
    # estimated_network = result_df
    #
    # return {'deepsem': estimated_network}
    #
    #


