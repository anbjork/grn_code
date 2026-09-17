


from pathlib import Path
import anton_util
import numpy as np

from grn_code.pipeline_code import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')

df = anton_util.unpickle_object(str(compiled_results_path))
anton_util.log_timestamp(f'data loaded, shape: {df.shape}')

outdir = Path(f'{output_base_path}/simulated/plots/')
outdir.mkdir(exist_ok=True, parents=True)

import matplotlib.pyplot as plt

def scatter(df, x, y, path):
    plt.close('all')
    plt.plot(df[x], df[y], '.', alpha=0.1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(path)

p = f'{output_base_path}/simulated/plots/auroc_vs_nTPs_with_perfect_all_genes.png'
scatter(df, 'n_TPs', 'AUROC', p)

from copy import deepcopy
df1 = deepcopy(df)
df = df.loc[df['method'] != 'perfect_inference_all_genes', :]

scatter(df, 'n_TPs', 'AUROC', 
        f'{output_base_path}/simulated/plots/auroc_vs_nTPs.png')
scatter(df, 'n_genes_after_harmonisation', 'AUROC', 
        f'{output_base_path}/simulated/plots/auroc_vs_n_genes.png')

counts, bins = np.histogram(df['n_genes_after_harmonisation'])
counts = np.log10(counts + 1)
plt.close('all')
plt.stairs(counts, bins, fill=True)
plt.savefig(f'{output_base_path}/simulated/plots/hist_ngenes.png')

low_ngenes = (df['n_genes_after_harmonisation'] < 30)
print(low_ngenes.sum())
low_ntps = (df['n_TPs'] < 60)
print(low_ntps.sum())
print((low_ngenes & low_ntps).sum())
# Output: all 80. Okay, so it's the datasets with few genes that have few ntps.
# Not unexpected

scatter(df, 'n_TPs', 'n_genes_after_harmonisation',
        f'{output_base_path}/simulated/plots/ntps_vs_n_genes.png')





