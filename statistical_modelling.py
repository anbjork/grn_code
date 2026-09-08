import statsmodels.formula.api as smf
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

CATEGORICAL_PREDICTORS = ['method', 'pseudo_bulk', 'cell normalised',
                           'transform 1', 'transform 2', 'replicate', 'control_delta']
CONTINUOUS_PREDICTORS  = ['snr', 'cell_count', 'n_TPs', 'n_genes_after_harmonisation',
                           'ERMA', '0_fraction__before_filtering__all']

print(f'Categorical predictors: {CATEGORICAL_PREDICTORS}')
print(f'Continuous predictors:  {CONTINUOUS_PREDICTORS}')

def needs_quoting(name):
    return ' ' in name or name[0].isdigit()

def term(name, categorical=False):
    q = f'Q("{name}")' if needs_quoting(name) else name
    return f'C({q})' if categorical else q

formula_terms = (
    [term(p, categorical=True) for p in CATEGORICAL_PREDICTORS] +
    [term(p, categorical=False) for p in CONTINUOUS_PREDICTORS]
)

for outcome in OUTCOMES:
    lhs = term(outcome)
    formula = lhs + ' ~ ' + ' + '.join(formula_terms)

    print('\n' + '=' * 72)
    print(f'Outcome : {outcome}')
    print(f'Formula : {formula}')
    print('=' * 72)

    cols = [outcome] + CATEGORICAL_PREDICTORS + CONTINUOUS_PREDICTORS
    df_model = df[cols].dropna()

    model = smf.ols(formula, data=df_model).fit()
    print(model.summary())

    anton_util.log_timestamp(f'done with {outcome}')

anton_util.log_timestamp('done')
