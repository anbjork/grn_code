
import numpy as np
import anton_util
import matplotlib.pyplot as plt
from local_imports import output_path



drop_methods = [
        'mean_difference_controls_only',
        'bigsm',
        'lsco',
        'perfect_inference_all_genes',
        'perfect_inference_filtered_genes',
        'zscore_ab',
        ]
verbose_dspin_label = 'dspin__use_perturbation_matrix_True__perturbation_knockdown_value_-0.9__use_prior_h_True'
method_label_translations = {
    verbose_dspin_label: 'D-SPIN',
    'lsco.T': 'LSCO',
    'zscore_dream3': 'Zscore',
    'bigsm_T': 'BiGSM',
    'correlation': 'Correlation',
    'deepsem': 'DeepSEM',
    'lasso_density_T_density_0.3': 'LASSO, density 0.3',
    'lasso_density_T_density_1': 'LASSO, density 1',
    'lasso_density_T_density_10': 'LASSO, density 10',
    'mean_difference': 'Mean Difference',
    'psgrn': 'PSGRN',
    'random': 'Random',
    'regdiffusion': 'RegDiffusion',
    }




anton_util.log_timestamp('plotting...')
anton_util.log_timestamp('reading data...')

df = anton_util.unpickle_object(
        f'{output_path}/simulated/compiled_results.pkl')
df['controlled_var'] = [elem.split(':')[0] for elem in df.data_case]

anton_util.log_timestamp('the rest...')


methods = set(df['method'])
for method in drop_methods:
    if method in methods:
        df = df[df['method'] != method]
for before, after in method_label_translations.items():
    if before in methods:
        df.loc[df['method'] == before, 'method'] = after


from pathlib import Path
output_dir = Path(f'{output_path}/simulated/plots')
output_dir.mkdir(exist_ok=True, parents=True)

dropout_cols = [
        '0_fraction__before_filtering__all',
        '0_fraction__after_filtering__all',
        ]
y_vars = ['AUROC', 'AUPR gain', 'top_k_accuracy']
# y_vars = ['AUROC']  # Debug

controlled_vars = {'snr', 'cell_count'}
dependent_vars = {'dispersion': dropout_cols}
xlims = {
        'snr': {
            (-0.01, 0.1),
            (-0.01, 0.2),
            (None, 1),
            (None, 2),
            }
        }
x_transforms = {
        'snr': [None, 'log'],
        'cell_count': [None, 'log'],
        }
x_vars = sorted(controlled_vars | set(dependent_vars.keys()))

def plot(y_var, x_var, x_transform, dependent_var):

    dfs = df[df['controlled_var'] == x_var]

    # Incredibly ugly
    if x_transform is not None and dependent_var is not None:
        raise ValueError('Cannot handle both x_transform and dependent_var set')

    if dependent_var is not None:
        x_disp = dependent_var
    else:
        x_disp = x_var

    if x_transform == 'log':
        x_disp = f'log10_{x_var}'
        dfs = dfs.copy()
        dfs[x_disp] = np.log10(dfs[x_var])
    elif x_transform is None:
        pass
    else:
        raise NotImplementedError(f'Unknown x_transform: {x_transform}')

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = set(dfs['method'])
    if np.nan in methods: methods.remove(np.nan)
    methods = sorted(methods)
    colors = plt.get_cmap('tab20').colors  # pyright: ignore
    for i, method in enumerate(methods):
        df_method = dfs[dfs['method'] == method]
        df_method = df_method.sort_values(by=x_disp)
        if x_var in controlled_vars:
            grouped = df_method.groupby(x_disp)[y_var]
            means = grouped.mean()
            sds = grouped.std()
            ax.errorbar(means.index, means, yerr=sds, label=method, capsize=3, color=colors[i % len(colors)])
        else:
            x = np.array(df_method[x_disp])
            smooth = np.array(df_method[y_var].rolling(window=15, center=True).mean())
            ax.plot(x, smooth, label=method, color=colors[i % len(colors)])
    # ax.set_xlim(0.75, 1)
    ax.set_xlabel(x_disp)
    ax.set_ylabel(y_var)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()

    out_base = f'{output_dir}/{y_var}_against_{x_disp}'
    fig.savefig(f'{out_base}.png')
    if x_var in xlims.keys() and x_transform is None:
        lim_tuples = xlims[x_var]
        for lim_tuple in lim_tuples:
            xlim = lim_tuple[1]
            ax.set_xlim(lim_tuple)  # pyright: ignore
            fig.savefig(f'{out_base}__xlim_{xlim}.png')


def dropouts_against_dispersion(dropout_col):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['dispersion'], df[dropout_col])
    out = f'{output_dir}/dropouts_against_dispersion__{dropout_col}'
    fig.savefig(f'{out}.png')
    ax.set_xlim(None, 5)
    fig.savefig(f'{out}_xlim.png')

def plot_wrapper_handle_dependent(y_var, x_var, x_transform):
    if x_var in dependent_vars.keys():
        for dependent_var in dependent_vars[x_var]:
            plot(y_var, x_var, x_transform, dependent_var)
            dropouts_against_dispersion(dependent_var)
    else:
        plot(y_var, x_var, x_transform, None)

for y_var in y_vars:
# for y_var in ['AUROC']:
    for x_var in x_vars:
        if x_var in x_transforms.keys():
            print(f'{x_var = }')
            for transform in x_transforms[x_var]:
                print(f'{transform = }')
                plot_wrapper_handle_dependent(y_var, x_var, transform)
        else:
            print(f'{x_var = }')
            plot_wrapper_handle_dependent(y_var, x_var, None)

anton_util.log_timestamp('done')







