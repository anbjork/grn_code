import matplotlib.pyplot as plt
import numpy as np
import anton_util
from pathlib import Path
from grn_code.pipeline_configuration import output_base_path

compiled_results_path = Path(f'{output_base_path}/simulated/compiled_results.pkl')
output_path = Path(f'{output_base_path}/simulated/plots/auroc_by_method.png')
output_path.parent.mkdir(parents=True, exist_ok=True)

df = anton_util.unpickle_object(str(compiled_results_path))

methods = sorted(df['method'].unique())
x_positions = {m: i for i, m in enumerate(methods)}

fig, ax = plt.subplots(figsize=(12, 6))

rng = np.random.default_rng(0)
for method, group in df.groupby('method'):
    x = x_positions[method]
    jitter = rng.uniform(-0.3, 0.3, size=len(group))
    ax.scatter(x + jitter, group['AUROC'], alpha=0.15, s=4, color='steelblue')
    ax.scatter(x, group['AUROC'].mean(), s=40, color='firebrick', zorder=5)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel('AUROC')
ax.set_title('AUROC by method')
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='random baseline (0.5)')
ax.legend()

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f'Saved to {output_path}')
