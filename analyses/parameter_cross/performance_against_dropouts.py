
import numpy as np
import anton_util
import matplotlib.pyplot as plt
from local_imports import output_path

anton_util.log_timestamp('plotting...')
anton_util.log_timestamp('reading data...')

df = anton_util.unpickle_object(
        f'{output_path}/simulated/compiled_results.pkl')

anton_util.log_timestamp('the rest...')

from pathlib import Path
output_dir = Path(f'{output_path}/simulated/plots')
output_dir.mkdir(exist_ok=True, parents=True)

dropout_col = '0_fraction__before_filtering__all'
y_vars = ['AUROC', 'AUPR gain', 'top_k_accuracy']

controlled_vars = {'snr', 'cell_count'}
dependent_vars = {'dispersion': dropout_col}
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
        }
x_vars = sorted(controlled_vars | set(dependent_vars.keys()))
df['controlled_var'] = [elem.split(':')[0] for elem in df.data_case]

def plot(y_var, x_var, x_transform):

    dfs = df[df['controlled_var'] == x_var]

    if x_var in dependent_vars.keys():
        x_disp = dependent_vars[x_var]
    else:
        x_disp = x_var

    if x_transform == 'log':
        x_disp = f'log10_{x_var}'
        dfs.loc[x_disp] = np.log10(dfs[x_var])
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
            smooth_auroc = np.array(df_method[y_var].rolling(window=10, center=True).mean())
            ax.plot(x, smooth_auroc, label=method, color=colors[i % len(colors)])
    # ax.set_xlim(0.75, 1)
    ax.set_xlabel(x_disp)
    ax.set_ylabel(y_var)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()

    out_base = f'{output_dir}/{y_var}_against_{x_disp}__transform_{x_transform}'
    fig.savefig(f'{out_base}.png')
    if x_var in xlims.keys() and x_transform is None:
        lim_tuples = xlims[x_var]
        for lim_tuple in lim_tuples:
            xlim = lim_tuple[1]
            ax.set_xlim(lim_tuple)  # pyright: ignore
            fig.savefig(f'{out_base}__xlim__{xlim}.png')

# for y_var in y_vars:
for y_var in ['AUROC']:
    for x_var in x_vars:
        if x_var in x_transforms.keys():
            print(f'{x_var = }')
            for transform in x_transforms[x_var]:
                print(f'{transform = }')
                plot(y_var, x_var, transform)
        else:
            print(f'{x_var = }')
            plot(y_var, x_var, None)



anton_util.log_timestamp('done')


plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['dispersion'], df[dropout_col])
fig.savefig(f'{output_dir}/dropouts_against_dispersion.png')
ax.set_xlim(None, 5)
fig.savefig(f'{output_dir}/dropouts_against_dispersion_xlim.png')






