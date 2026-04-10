import anton_util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from plot_functions import create_auroc_plots, create_aupr_erma_plots, create_aupr_ratio_plots

# Create output directory
os.makedirs('./plots/replogle', exist_ok=True)

# Load the data
df = anton_util.unpickle_object('benchmarks/replogle/stats_df.pkl')

# Filter for specific methods
methods_to_plot = ['lsco', 'lsco.T', 'zscore_ab', 'zscore_dream3']
df_filtered = df[df['method'].isin(methods_to_plot)].copy()

# Convert pseudo_bulk to string for better plotting
df_filtered['pseudo_bulk_str'] = df_filtered['pseudo_bulk'].astype(str)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

print(f"Total rows after filtering: {len(df_filtered)}")
print("Data points per method and pseudo_bulk category:")
for method in methods_to_plot:
    method_data = df_filtered[df_filtered['method'] == method]
    print(f"\n{method}:")
    counts = method_data['pseudo_bulk'].value_counts().sort_index()
    print(counts)


# AUROC plot
create_auroc_plots(df_filtered, methods_to_plot, 'Replogle Data', './plots/replogle/auroc_comparison_scatter.png')

# AUPR and ERMA plot
create_aupr_erma_plots(df_filtered, methods_to_plot, 'Replogle Data', './plots/replogle/aupr_erma_comparison_scatter.png')

# AUPR ratio plot
create_aupr_ratio_plots(df_filtered, methods_to_plot, 'Replogle Data', './plots/replogle/aupr_ratio_comparison_scatter.png')

# Plots for specific reference networks
reference_networks = [
    'Non-specific-ChIP-seq-network_with_weights',
    'k562_cistrome_regp_geq_3.00.csv'
]

for ref_network in reference_networks:
    # Filter data for this reference network
    df_ref = df_filtered[df_filtered['reference_network'] == ref_network].copy()
    
    if len(df_ref) == 0:
        print(f"No data found for reference network: {ref_network}")
        continue
    
    # Clean reference network name for filename
    ref_name_clean = ref_network.replace('.csv', '')
    
    # AUROC plot for this reference network
    create_auroc_plots(df_ref, methods_to_plot, f'Replogle Data ({ref_network})', 
                      f'./plots/replogle/auroc_comparison_{ref_name_clean}_scatter.png')
    
    # AUPR and ERMA plot for this reference network
    create_aupr_erma_plots(df_ref, methods_to_plot, f'Replogle Data ({ref_network})', 
                          f'./plots/replogle/aupr_erma_comparison_{ref_name_clean}_scatter.png')
    
    # AUPR ratio plot for this reference network
    create_aupr_ratio_plots(df_ref, methods_to_plot, f'Replogle Data ({ref_network})', 
                           f'./plots/replogle/aupr_ratio_comparison_{ref_name_clean}_scatter.png')

# Print summary statistics
print("\nSummary Statistics by Method and Pseudo-bulk Category:")
print("=" * 60)
for method in methods_to_plot:
    method_data = df_filtered[df_filtered['method'] == method]
    print(f"\n{method.upper()}:")
    summary = method_data.groupby('pseudo_bulk_str')[['AUROC', 'AUPR', 'ERMA', 'AUPR ratio']].agg(['mean', 'std'])
    print(summary.round(4))
