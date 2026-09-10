import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.diagnostic import linear_reset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import numpy as np
import pandas as pd
from pathlib import Path
import anton_util

from grn_code.pipeline_configuration import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')
output_dir = Path(f'{output_base_path}/simulated/statistical_modelling')

df = anton_util.unpickle_object(str(compiled_results_path))
# Excluded: perfect_inference_all_genes operates on unfiltered genes, giving it
# artificially high n_TPs and AUROC=1, making it incomparable to other methods.
# It also formed a separate cluster in n_TPs and n_genes_after_harmonisation,
# which distorted the effect estimates for those predictors before exclusion.
# Excluded: perfect_inference_filtered_genes always achieves AUROC=1 by construction,
# creating a ceiling effect and influential outliers in the residuals vs fitted plot.
# Excluded: lsco appears to not work correctly, excluded to avoid biasing results.
df = df[~df['method'].isin(['perfect_inference_all_genes', 'perfect_inference_filtered_genes', 'lsco'])].copy()
anton_util.log_timestamp(f'data loaded, shape: {df.shape}')
print(f'Columns available: {list(df.columns)}')

# OUTCOMES = ['AUPR ratio', 'AUPR', 'AUROC', 'top_k_accuracy']
OUTCOMES = ['AUROC']

CATEGORICAL_PREDICTORS = {
    'method':                    'random',
    'pseudo_bulk':               'False',
    'cell normalised':           'False',
    'transform 1':               'none',
    'transform 2':               'none',
    'control_delta':             'False',
    'replicate':                 '0',
}
CONTINUOUS_PREDICTORS = ['snr', 'cell_count', 'n_genes_after_harmonisation', 'ERMA',
                          '0_fraction__before_filtering__all']

# n_TPs removed: highly correlated with both n_genes_after_harmonisation (r~0.86) and ERMA (VIF ~80).
# Structurally redundant — n_TPs counts edges in the reference network, ERMA is fully determined
# by the same network density, and n_genes is more interpretable as a data property.
# n_genes and ERMA are not strongly correlated with each other, so both are retained.
# Before removal, including all three caused sign flips on ERMA (positive in univariate screen,
# negative in joint model), a classic symptom of multicollinearity.

SIGNIFICANCE_THRESHOLD = 0.05

# Cast categorical predictors to string to avoid bool/int coercion issues
for col in CATEGORICAL_PREDICTORS:
    df[col] = df[col].astype(str)

def print_distributions(cols, header):
    print(f'\n{header} (min, Q25, median, Q75, max):')
    for col in cols:
        lo, q25, med, q75, hi = df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
        print(f'  {col}: {lo:.3g}, {q25:.3g}, {med:.3g}, {q75:.3g}, {hi:.3g}')

# Scale continuous predictors by 2 standard deviations (mean ± 1 std).
# Coefficients then represent the effect of going from mean-1sd to mean+1sd.
df_model_base = df.copy()

print('\nLevels of categorical predictors:')
for col, ref_level in CATEGORICAL_PREDICTORS.items():
    levels = sorted(df[col].unique())
    print(f'  {col}: {levels}  (reference: {ref_level})')
zero_fraction_cols = [c for c in df.columns if '0_fraction' in c]
print_distributions(zero_fraction_cols, 'Zero fraction variable distributions')

print_distributions(CONTINUOUS_PREDICTORS, 'Continuous predictor distributions')

print('\nContinuous predictor scaling (mean, sd, mean-1sd, mean+1sd):')
for col in CONTINUOUS_PREDICTORS:
    mean = df[col].mean()
    sd = df[col].std()
    if sd == 0:
        raise ValueError(f'SD is zero for {col!r} — cannot scale.')
    lo, hi = mean - sd, mean + sd
    print(f'  {col}: mean={mean:.3g}, sd={sd:.3g}, range=[{lo:.3g}, {hi:.3g}]')
    df_model_base[col] = (df[col] - mean) / (2 * sd)

print(f'\nContinuous predictors (±1sd-scaled): {CONTINUOUS_PREDICTORS}')

def needs_quoting(name):
    return ' ' in name or name[0].isdigit()

def ref(name):
    """Return a formula reference to a column, quoting if necessary."""
    return f'Q("{name}")' if needs_quoting(name) else name

def cat_term(name, ref_level):
    """Categorical term with explicit reference level."""
    return f'C({ref(name)}, Treatment("{ref_level}"))'

def cont_term(name):
    """Continuous term."""
    return ref(name)

def joint_f_pvalue(model, term_prefix):
    """F-test p-value for all levels of a categorical term."""
    matching = [t for t in model.params.index if t.startswith(term_prefix)]
    if not matching:
        return float('nan')
    return model.f_test([f'{t} = 0' for t in matching]).pvalue

for outcome in OUTCOMES:
    lhs = ref(outcome)

    # --- Screening: fit full model, drop insignificant predictors ---
    all_terms = (
        [cat_term(col, ref_level) for col, ref_level in CATEGORICAL_PREDICTORS.items()] +
        [cont_term(p) for p in CONTINUOUS_PREDICTORS]
    )
    screen_formula = lhs + ' ~ ' + ' + '.join(all_terms)
    cols_all = [outcome] + list(CATEGORICAL_PREDICTORS.keys()) + CONTINUOUS_PREDICTORS
    df_model = df_model_base[cols_all].dropna()
    screen_model = smf.ols(screen_formula, data=df_model).fit()

    print('\n' + '=' * 72)
    print(f'Screening stage for outcome: {outcome}')
    print(f'  {"Predictor":<45} {"p-value":>10}  {"included?":>10}')
    print(f'  {"-"*45} {"-"*10}  {"-"*10}')

    active_cat = {}
    for col, ref_level in CATEGORICAL_PREDICTORS.items():
        prefix = f'C({ref(col)}, Treatment("{ref_level}"))'
        pval = joint_f_pvalue(screen_model, prefix)
        keep = pval < SIGNIFICANCE_THRESHOLD
        if keep:
            active_cat[col] = ref_level
        print(f'  {col:<45} {pval:>10.4f}  {"yes" if keep else "no":>10}')

    active_cont = []
    for col in CONTINUOUS_PREDICTORS:
        term = cont_term(col)
        pval = screen_model.pvalues.get(term, float('nan'))
        keep = pval < SIGNIFICANCE_THRESHOLD
        if keep:
            active_cont.append(col)
        print(f'  {col:<45} {pval:>10.4f}  {"yes" if keep else "no":>10}')

    # --- Main model: refit with only significant predictors ---
    formula_terms_active = (
        [cat_term(col, ref_level) for col, ref_level in active_cat.items()] +
        [cont_term(p) for p in active_cont]
    )

    formula = lhs + ' ~ ' + ' + '.join(formula_terms_active)

    print('\n' + '=' * 72)
    print(f'Outcome : {outcome}')
    print(f'Formula : {formula}')
    print('=' * 72)

    model = smf.ols(formula, data=df_model).fit()
    print(model.summary())

    # --- Effect size summary ---
    # Each row is one coefficient (categorical levels individually, continuous as-is).
    # Categorical coefficients are relative to their reference level.
    # Continuous coefficients represent the effect of ±1sd around the mean.
    rows = []
    for col, ref_level in active_cat.items():
        prefix = f'C({ref(col)}, Treatment("{ref_level}"))'
        for term, coef in model.params.items():
            if not term.startswith(prefix):
                continue
            ci = model.conf_int().loc[term]
            pval = model.pvalues[term]
            # Extract the level name from the term string, e.g. "[T.True]" -> "True"
            level = term[len(prefix):]
            if level.startswith('[T.') and level.endswith(']'):
                level = level[3:-1]
            rows.append({
                'predictor': f'{col} = {level}',
                'abs_coef': abs(coef),
                'coef': coef,
                'ci_lo': ci[0],
                'ci_hi': ci[1],
                'pval': pval,
            })
    for col in active_cont:
        term = cont_term(col)
        coef = model.params[term]
        ci = model.conf_int().loc[term]
        pval = model.pvalues[term]
        rows.append({
            'predictor': col,
            'abs_coef': abs(coef),
            'coef': coef,
            'ci_lo': ci[0],
            'ci_hi': ci[1],
            'pval': pval,
        })
    # Also include interaction terms not captured above
    main_prefixes = tuple(
        f'C({ref(col)}, Treatment("{ref_level}"))' for col, ref_level in active_cat.items()
    ) + tuple(cont_term(p) for p in active_cont)
    for term, coef in model.params.items():
        if term == 'Intercept':
            continue
        if any(term.startswith(p) and ':' not in term for p in main_prefixes):
            continue
        if ':' in term:
            ci = model.conf_int().loc[term]
            pval = model.pvalues[term]
            rows.append({
                'predictor': term,
                'abs_coef': abs(coef),
                'coef': coef,
                'ci_lo': ci[0],
                'ci_hi': ci[1],
                'pval': pval,
            })

    rows.sort(key=lambda r: r['abs_coef'], reverse=True)

    print('\nEffect size ranking (categorical levels vs reference, continuous ±1sd):')
    print(f'  {"predictor":<55} {"coef":>8}  {"95% CI"}               {"p-value":>10}')
    print(f'  {"-"*55} {"-"*8}  {"-"*20}  {"-"*10}')
    for r in rows:
        print(f'  {r["predictor"]:<55} {r["coef"]:>8.4f}  [{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]  {r["pval"]:>10.4f}')

    # --- Coefficient forest plot ---
    # Plot all individual coefficients (excluding intercept), sorted by |coef|
    params = model.params.drop('Intercept')
    ci = model.conf_int().drop('Intercept')
    order = params.abs().sort_values(ascending=True).index
    params = params[order]
    ci = ci.loc[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.3)))
    y = range(len(params))
    ax.barh(list(y), params.values, xerr=[params.values - ci[0].values, ci[1].values - params.values],
            align='center', height=0.6, color='steelblue', ecolor='black', capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(params.index, fontsize=7)
    ax.set_xlabel('Coefficient (AUROC scale, continuous predictors ±1sd-scaled)')
    ax.set_title(f'Effect sizes: {outcome}')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'effect_sizes_{outcome}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    # R² per method subgroup
    print('\nR² per method subgroup:')
    for method, grp in df_model.groupby('method'):
        y = grp[outcome]
        y_hat = model.fittedvalues.loc[grp.index]
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f'  {method:<40} R²={r2:.3f}  n={len(grp)}')

    print(f'\nCoefficient plot saved to {plot_path}')

    # --- Diagnostics ---
    fitted = model.fittedvalues
    residuals = model.resid
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    std_resid = influence.resid_studentized_internal

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Residuals vs fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, residuals, alpha=0.2, s=3)
    ax1.axhline(0, color='red', linewidth=0.8)
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')

    # Q-Q plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, plot=ax2)
    ax2.set_title('Q-Q plot of residuals')

    # Scale-location
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.2, s=3)
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('√|Standardised residuals|')
    ax3.set_title('Scale-Location')

    # Leverage vs residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(leverage, std_resid, alpha=0.2, s=3)
    ax4.axhline(0, color='red', linewidth=0.8)
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardised residuals')
    ax4.set_title('Leverage vs Residuals')

    # Cook's distance
    ax5 = fig.add_subplot(gs[1, 1])
    cooks_threshold = 4 / len(cooks_d)
    ax5.stem(range(len(cooks_d)), cooks_d, markerfmt=',', linefmt='grey', basefmt='k-')
    ax5.axhline(cooks_threshold, color='red', linewidth=0.8, linestyle='--',
                label=f'4/n={cooks_threshold:.4f}')
    ax5.legend(fontsize=7)
    ax5.set_xlabel('Observation index')
    ax5.set_ylabel("Cook's distance")
    ax5.set_title("Cook's Distance")

    # VIF
    ax6 = fig.add_subplot(gs[1, 2])
    vif_terms = [t for t in model.model.exog_names if t != 'Intercept']
    vif_vals = [oi.variance_inflation_factor(model.model.exog, i)
                for i, t in enumerate(model.model.exog_names) if t != 'Intercept']
    vif_y = range(len(vif_terms))
    ax6.barh(list(vif_y), vif_vals, color='steelblue')
    ax6.axvline(5, color='orange', linewidth=0.8, linestyle='--', label='VIF=5')
    ax6.axvline(10, color='red', linewidth=0.8, linestyle='--', label='VIF=10')
    ax6.set_yticks(list(vif_y))
    ax6.set_yticklabels(vif_terms, fontsize=6)
    ax6.set_xlabel('VIF')
    ax6.set_title('Variance Inflation Factors')
    ax6.legend(fontsize=7)

    plt.suptitle(f'Diagnostics: {outcome}', fontsize=12)
    plt.tight_layout()
    diag_path = output_dir / f'diagnostics_{outcome}.png'
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f'Diagnostics plot saved to {diag_path}')

    # RESET test for functional form misspecification
    # Tests whether adding powers of fitted values improves the model (significant = misspecified)
    reset_result = linear_reset(model, power=3, use_f=True)
    print(f'\nRESET test (powers 2-3 of fitted values):')
    print(f'  F-statistic: {reset_result.statistic:.4f}')
    print(f'  p-value:     {reset_result.pvalue:.4f}')
    print(f'  Interpretation: {"misspecification likely" if reset_result.pvalue < 0.05 else "no evidence of misspecification"}')

    # Top outliers by Cook's distance — print full row from original df for inspection
    n_top = 10
    top_idx = np.argsort(cooks_d)[-n_top:][::-1]
    print(f"\nTop {n_top} observations by Cook's distance (threshold=4/n={cooks_threshold:.4f}):")
    print(f'  {"idx":>6}  {"cook_d":>10}  {"std_resid":>10}  {"fitted":>8}  {"actual":>8}')
    for i in top_idx:
        print(f'  {i:>6}  {cooks_d[i]:>10.5f}  {std_resid[i]:>10.3f}  '
              f'{fitted.iloc[i]:>8.3f}  {df_model[outcome].iloc[i]:>8.3f}')
    top_rows = df.iloc[top_idx].copy()
    top_rows.insert(0, 'cook_d', [cooks_d[i] for i in top_idx])
    top_rows.insert(1, 'std_resid', [std_resid[i] for i in top_idx])
    outlier_path = output_dir / f'outliers_{outcome}.pkl'
    anton_util.pickle_object(top_rows, str(outlier_path))
    print(f'Outlier rows saved to {outlier_path}')

    # Cook's D vs n_genes and n_TPs to check connection between outliers and low gene counts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ['n_genes_after_harmonisation', 'n_TPs']):
        ax.scatter(df.loc[df_model.index, col], cooks_d, alpha=0.2, s=3)
        ax.axhline(cooks_threshold, color='red', linewidth=0.8, linestyle='--',
                   label=f'4/n={cooks_threshold:.4f}')
        ax.set_xlabel(col)
        ax.set_ylabel("Cook's D")
        ax.set_title(f"Cook's D vs {col}")
        ax.legend(fontsize=7)
    plt.suptitle(f"Cook's D vs gene/TP counts: {outcome}", fontsize=10)
    plt.tight_layout()
    cooks_genes_path = output_dir / f'cooks_d_vs_gene_counts_{outcome}.png'
    plt.savefig(cooks_genes_path, dpi=150)
    plt.close()
    print(f"Cook's D vs gene counts plot saved to {cooks_genes_path}")

    # Residuals vs each predictor (to diagnose pattern sources)
    all_pred_cols = list(active_cat.keys()) + active_cont
    ncols = 3
    nrows = int(np.ceil(len(all_pred_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for ax, col in zip(axes, all_pred_cols):
        ax.scatter(df_model[col].astype(str) if col in active_cat else df_model[col],
                   residuals, alpha=0.15, s=3)
        ax.axhline(0, color='red', linewidth=0.8)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel('Residual', fontsize=8)
        ax.set_title(col, fontsize=8)
        if col in active_cat:
            ax.tick_params(axis='x', labelrotation=30, labelsize=6)
    for ax in axes[len(all_pred_cols):]:
        ax.set_visible(False)
    plt.suptitle(f'Residuals vs predictors: {outcome}', fontsize=10)
    plt.tight_layout()
    resid_pred_path = output_dir / f'residuals_vs_predictors_{outcome}.png'
    plt.savefig(resid_pred_path, dpi=150)
    plt.close()
    print(f'Residuals vs predictors plot saved to {resid_pred_path}')

    # Predictor correlation heatmap (continuous predictors only)
    cont_cols = active_cont
    if len(cont_cols) > 1:
        corr = df_model[cont_cols].corr()
        fig, ax = plt.subplots(figsize=(max(4, len(cont_cols)), max(3, len(cont_cols))))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(cont_cols)))
        ax.set_yticks(range(len(cont_cols)))
        ax.set_xticklabels(cont_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(cont_cols, fontsize=8)
        for i in range(len(cont_cols)):
            for j in range(len(cont_cols)):
                ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Continuous predictor correlations: {outcome}')
        plt.tight_layout()
        corr_path = output_dir / f'predictor_correlations_{outcome}.png'
        plt.savefig(corr_path, dpi=150)
        plt.close()
        print(f'Correlation heatmap saved to {corr_path}')

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
