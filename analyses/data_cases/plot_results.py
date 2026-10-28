import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# METRICS = ['AUROC', 'AUPR', 'AUPR ratio', 'AUPR gain', 'top_k_accuracy']
METRICS = ['AUROC', 'AUPR gain', 'top_k_accuracy', 'precision_at_k']
DROPOUT_COL = '0_fraction__before_filtering__all'
VARS_TO_STRATIFY = [
    'cell normalised',
    # 'read normalised',
    'transform 1',
    'transform 2',
    'pseudo_bulk',
    'control_delta',
]
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



def load_df(output_path):
    import anton_util
    anton_util.log_timestamp('reading data...')
    df = anton_util.unpickle_object(
            f'{output_path}/simulated/compiled_results.pkl')
    df = df.loc[df['inference error'].isna(), :]
    return df


def label_data_cases(df):
    dropout = df.groupby('data_case')[DROPOUT_COL].mean()
    df = df.copy()
    df['data_case'] = [f'{dc} | dropout {dropout[dc]:.2f}' for dc in df['data_case']]
    return df


def to_long(df):
    id_cols = [c for c in df.columns if c not in METRICS]
    return df.melt(id_vars=id_cols, value_vars=METRICS,
                   var_name='metric', value_name='value')


def get_configs(df_long, major_col, minor_col):
    exclude = {major_col, minor_col, 'method', 'value'}
    vars_to_stratify = [c for c in VARS_TO_STRATIFY
                        if c in df_long.columns and c not in exclude]
    options = {v: df_long[v].unique() for v in vars_to_stratify}
    configs = []
    def recursive_combos(determined, remaining):
        from copy import deepcopy
        if len(remaining) == 0:
            configs.append(deepcopy(determined))
            return
        k, opts = remaining.popitem()
        for option in opts:
            determined[k] = option
            recursive_combos(determined, deepcopy(remaining))
    recursive_combos({}, options)
    return configs


def subset_df(df, config):
    from copy import deepcopy
    df_subset = deepcopy(df)
    for var, option in config.items():
        df_subset = df_subset[df_subset[var] == option]
    return df_subset


def plot_metrics(output_path, major_col, minor_col):
    """
    One figure per value of major_col, one subplot per value of minor_col,
    methods on x-axis.
    """
    import anton_util
    anton_util.log_timestamp('plotting...')

    df_raw = load_df(output_path)
    # df_raw = df_raw.loc[df_raw['method'] != 'mean_difference_controls_only']

    # pretty assumes that all the methods for paper are present. Turn it off if not
    pretty = True
    if pretty:
        for method in drop_methods:
            df_raw = df_raw.loc[df_raw['method'] != method, :]
        for before, after in method_label_translations.items():
            df_raw.loc[df_raw['method'] == before, 'method'] = after

    df_long = to_long(label_data_cases(df_raw))
    output_dir = Path(output_path) / 'simulated' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    anton_util.log_timestamp('separating by preprocessing...')
    configs = get_configs(df_long, major_col, minor_col)
    print('configs:')
    for config in configs:
        print(config)

    for config in configs:
        df_config = subset_df(df_long, config)
        config_name = ' | '.join([f'{k} {v}' for k, v in config.items()])
        for major_val in df_config[major_col].unique():
            df_major = df_config[df_config[major_col] == major_val]
            plot_name = f'{major_val} | {config_name}'
            anton_util.log_timestamp(f'{plot_name}...')
            fig = _make_plot(df_major, subplot_col=minor_col)
            fig.text(0.5, 1.02, major_val, ha='center', fontsize=12, fontweight='bold')
            fig.text(0.5, 0.97, config_name, ha='center', fontsize=12)
            fig.savefig(output_dir / f'{plot_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

    anton_util.log_timestamp('picking preferred preprocessings...')
    not_zscores = [
            'D-SPIN',
            ]
    zscores = list(set(df_long['method']) - set(not_zscores))
    transforms2 = ['none', 'zscores']
    dfs = []
    for t2, methods in zip(transforms2, [not_zscores, zscores]):
        for method in methods:
            df = df_long.loc[df_long['method'] == method, :]
            df = df.loc[df['transform 2'] == t2, :]
            dfs.append(df)
    import pandas as pd
    df_preferred = pd.concat(dfs, axis=0)

    for major_val in df_preferred[major_col].unique():
        df_major = df_preferred[df_preferred[major_col] == major_val]
        plot_name = f'{major_val} | preferred preprocessings'
        anton_util.log_timestamp(f'{plot_name}...')
        fig = _make_plot(df_major, subplot_col=minor_col)
        # fig.text(0.5, 1.02, major_val, ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 1, major_val, ha='center', fontsize=12, fontweight='bold')
        # fig.text(0.5, 0.97, config_name, ha='center', fontsize=12)
        fig.savefig(output_dir / f'{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()



    anton_util.log_timestamp('plotting done')


def _make_plot(df_long, subplot_col):
    subplot_values = df_long[subplot_col].unique()
    methods = sorted(df_long['method'].unique())
    colors = plt.cm.tab20.colors  # pyright: ignore
    x_positions = np.arange(len(methods))

    fig, axes = plt.subplots(1, len(subplot_values),
                             figsize=(5 * len(subplot_values), 5),
                             squeeze=False)
    axes = axes[0]
    plt.subplots_adjust(bottom=0.4)

    for i, (ax, subplot_val) in enumerate(zip(axes, subplot_values)):
        df_sub = df_long[df_long[subplot_col] == subplot_val].copy()
        y_max = df_sub['value'].max()
        for j, method in enumerate(methods):
            values = df_sub[df_sub['method'] == method]['value']
            jitter = np.random.normal(0, 0.1, len(values))
            ax.scatter(x_positions[j] + jitter, values,
                       alpha=0.7, s=50, color=colors[j])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        # Small labels when crowded
        # ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=6)
        ax.set_title(f'{subplot_val}')
        # ax.set_ylim(-0.1, y_max * 1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('value')

    return fig




