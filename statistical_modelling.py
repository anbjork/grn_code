import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from pathlib import Path
import anton_util

from grn_code.pipeline_configuration import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')
output_dir = Path(f'{output_base_path}/simulated/statistical_modelling')

df = anton_util.unpickle_object(str(compiled_results_path))
anton_util.log_timestamp(f'data loaded, shape: {df.shape}')
print(f'Columns available: {list(df.columns)}')

# OUTCOMES = ['AUPR ratio', 'AUPR', 'AUROC', 'top_k_accuracy']
OUTCOMES = ['AUROC']  # Debug

CATEGORICAL_PREDICTORS = {
    'method':           'random',
    'pseudo_bulk':      'False',
    'cell normalised':  'False',
    'transform 1':      'none',
    'transform 2':      'none',
    'replicate':        '0',
    'control_delta':    'False',
}
CONTINUOUS_PREDICTORS = ['snr', 'cell_count', 'n_TPs', 'n_genes_after_harmonisation',
                          'ERMA', '0_fraction__before_filtering__all']

# Cast categorical predictors to string to avoid bool/int coercion issues
for col in CATEGORICAL_PREDICTORS:
    df[col] = df[col].astype(str)

# Standardize continuous predictors so coefficients are in units of 1 SD
df_model_base = df.copy()
print('\nContinuous predictor scaling (mean, std):')
for col in CONTINUOUS_PREDICTORS:
    mean, std = df[col].mean(), df[col].std()
    print(f'  {col}: mean={mean:.3g}, std={std:.3g}')
    df_model_base[col] = (df[col] - mean) / std

print('\nLevels of categorical predictors:')
for col, ref_level in CATEGORICAL_PREDICTORS.items():
    levels = sorted(df[col].unique())
    print(f'  {col}: {levels}  (reference: {ref_level})')

print(f'\nContinuous predictors (standardized): {CONTINUOUS_PREDICTORS}')

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

formula_terms = (
    [cat_term(col, ref_level) for col, ref_level in CATEGORICAL_PREDICTORS.items()] +
    [cont_term(p) for p in CONTINUOUS_PREDICTORS]
)

for outcome in OUTCOMES:
    lhs = ref(outcome)
    formula = lhs + ' ~ ' + ' + '.join(formula_terms)

    print('\n' + '=' * 72)
    print(f'Outcome : {outcome}')
    print(f'Formula : {formula}')
    print('=' * 72)

    cols = [outcome] + list(CATEGORICAL_PREDICTORS.keys()) + CONTINUOUS_PREDICTORS
    df_model = df_model_base[cols].dropna()

    model = smf.ols(formula, data=df_model).fit()
    print(model.summary())

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
