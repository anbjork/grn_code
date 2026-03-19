# From
# Large-scale causal discovery using interventional data sheds light on gene network structure in k562 cells

import urllib.request
from pathlib import Path

gene_excel_url = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-64353-7/MediaObjects/41467_2025_64353_MOESM4_ESM.xlsx'
gene_excel_path = Path('data/replogle/gene_set_from_tuuli.xlsx')

print(f'Downloading gene list excel file to {gene_excel_path} ...')
gene_excel_path.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(gene_excel_url, gene_excel_path)
print('Download complete.')



