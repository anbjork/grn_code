import anton_util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# REPLOGLE DATA (COMMENTED OUT)
# # Load the actual data
# df = anton_util.unpickle_object('benchmarks/replogle/stats_df.pkl')

# # Set pandas display options
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)

# # Clean up the dataframe
# df = df.drop('f1_scores', axis=1)
# df = df.rename({'n_genes_after_harmonisation': 'n_genes'}, axis=1)
# df = df.sort_values(by=['transform', 'method'])
# df = df.reset_index(drop=True)

# print("Data loaded and processed:")
# print(df.head())

# # Filter data according to specifications
# filtered_df = df[
#     (df['transform'] == 'log1p') & 
#     (df['pseudo_bulk'] == True) & 
#     (df['delta'] == False)
# ]

# # Create separate datasets for the two reference types
# nonspecific_df = filtered_df[filtered_df['reference'] == 'Non-specific-ChIP-seq-network_with_weights']
# k562_df = filtered_df[filtered_df['reference'] == 'k562_cistrome_regp_geq_3.00.csv']

# # Create the plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Plot 1: Non-specific ChIP-seq network
# if not nonspecific_df.empty:
#     methods = nonspecific_df['method'].unique()
#     colors = plt.cm.tab10(range(len(methods)))
#     for i, method in enumerate(methods):
#         method_data = nonspecific_df[nonspecific_df['method'] == method]
#         ax1.scatter(method_data['AUPR ratio'], method_data['AUROC'], 
#                    alpha=0.7, s=60, color=colors[i], label=method)
#     ax1.set_xlabel('AUPR ratio')
#     ax1.set_ylabel('AUROC')
#     ax1.set_title('Non-specific ChIP-seq Network')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

# # Plot 2: K562 filtered data
# if not k562_df.empty:
#     methods = k562_df['method'].unique()
#     colors = plt.cm.tab10(range(len(methods)))
#     for i, method in enumerate(methods):
#         method_data = k562_df[k562_df['method'] == method]
#         ax2.scatter(method_data['AUPR ratio'], method_data['AUROC'], 
#                    alpha=0.7, s=60, color=colors[i], label=method)
#     ax2.set_xlabel('AUPR ratio')
#     ax2.set_ylabel('AUROC')
#     ax2.set_title('K562 Cistrome (regp filtered)')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()

# plt.tight_layout()

# # Create output directory if it doesn't exist
# os.makedirs('outputs_plot_shit', exist_ok=True)

# # Save the plot as SVG
# plt.savefig('outputs_plot_shit/auroc_vs_aupr_ratio.svg', format='svg', bbox_inches='tight')
# plt.close()

# # Print summary statistics
# print("Filtered data summary:")
# print(f"Total filtered rows: {len(filtered_df)}")
# print(f"Non-specific network rows: {len(nonspecific_df)}")
# print(f"K562 rows: {len(k562_df)}")

# print("\nNon-specific ChIP-seq network data (all columns):")
# print(nonspecific_df)

# print("\nK562 cistrome data (all columns):")
# print(k562_df)

# print("\nUnique methods in K562 data:")
# for method in k562_df['method'].unique():
#     method_data = k562_df[k562_df['method'] == method]
#     print(f"\n{method} ({len(method_data)} rows) - all columns:")
#     print(method_data)

# SIMULATED DATA
# Load the simulated data
df_sim = anton_util.unpickle_object('benchmarks/simulated/stats_df.pkl')

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Clean up the dataframe (may need adjustment based on simulated data structure)
if 'f1_scores' in df_sim.columns:
    df_sim = df_sim.drop('f1_scores', axis=1)
if 'n_genes_after_harmonisation' in df_sim.columns:
    df_sim = df_sim.rename({'n_genes_after_harmonisation': 'n_genes'}, axis=1)
df_sim = df_sim.sort_values(by=['transform', 'method'])
df_sim = df_sim.reset_index(drop=True)

# # Fix the delta column naming bug by merging control_delta into delta
# df_sim['delta'] = df_sim['delta'].fillna(df_sim['control_delta'])

print("Simulated data loaded and processed:")
print(df_sim.head())
print("\nSimulated data columns:")
print(df_sim.columns.tolist())
print("\nSimulated data shape:")
print(df_sim.shape)
print("\nAvailable pseudo_bulk values:")
print(df_sim['pseudo_bulk'].unique())

# Filter simulated data according to specifications
filtered_df_sim = df_sim[
    (df_sim['transform'] == 'log1p')
    & (df_sim['pseudo_bulk'] == 10)
    # & (df_sim['delta'] == False)
    # & (df_sim['is_shuffled'] == False)
]

print(f"\nFiltered simulated data shape: {filtered_df_sim.shape}")
print("Filtered simulated data head:")
print(filtered_df_sim.head())

# Create the simulated data plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

if not filtered_df_sim.empty:
    methods = filtered_df_sim['method'].unique()
    colors = plt.cm.tab10(range(len(methods)))
    for i, method in enumerate(methods):
        method_data = filtered_df_sim[filtered_df_sim['method'] == method]
        ax.scatter(method_data['AUPR ratio'], method_data['AUROC'], 
                   alpha=0.7, s=60, color=colors[i], label=method)
    ax.set_xlabel('AUPR ratio')
    ax.set_ylabel('AUROC')
    ax.set_title('Simulated Data - AUROC vs AUPR ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()

plot_dir = 'plots'
# Create output directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

# Save the simulated plot as PNG
plt.savefig(f'{plot_dir}/auroc_vs_aupr_ratio_simulated.png', format='png', bbox_inches='tight', dpi=300)
plt.close()

# Print summary statistics for simulated data
print(f"\nSimulated data summary:")
print(f"Total filtered rows: {len(filtered_df_sim)}")
print(f"Unique methods: {filtered_df_sim['method'].unique()}")
print(f"Unique replicates: {filtered_df_sim['replicate'].unique()}")

print("\nFiltered simulated data (all columns):")
print(filtered_df_sim)




