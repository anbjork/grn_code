import matplotlib.pyplot as plt
import numpy as np
import anton_util
from pathlib import Path
from grn_code.pipeline_code import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')
output_path = Path(f'{output_base_path}/simulated/plots/auroc_by_method.png')
output_path.parent.mkdir(parents=True, exist_ok=True)

df = anton_util.unpickle_object(str(compiled_results_path))

sm = np.array(list(set(df['method'])))
tmp = ['psgrn_inference' in col for col in sm]
psgrn_errors_func_name = sm[tmp][0]
ms = [
        'zscore_ab_without_controls', 
        'perfect_inference_all_genes',
        psgrn_errors_func_name,
        ]
cs = [df['method'] != m for m in ms]
cs2 = np.array([list(c) for c in cs]).T
combined = cs2.all(axis=1)
df = df[combined]

yvar = 'AUROC'
methods = sorted(df['method'].unique())
x_positions = {m: i for i, m in enumerate(methods)}

fig, ax = plt.subplots(figsize=(12, 6))

rng = np.random.default_rng(0)
for method, group in df.groupby('method'):
    x = x_positions[method]
    jitter = rng.uniform(-0.3, 0.3, size=len(group))
    ax.scatter(x + jitter, group[yvar], alpha=0.15, s=4, color='steelblue')
    ax.scatter(x, group[yvar].mean(), s=40, color='firebrick', zorder=5)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel(yvar)
ax.set_title(f'{yvar} by method')
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='random baseline (0.5)')
ax.legend()

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f'Saved to {output_path}')
