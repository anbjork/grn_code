import json
import requests
from tqdm import tqdm

response = requests.get('https://api.figshare.com/v2/articles/20029387')
response.raise_for_status()
meta_data = response.json()

with open('data/replogle/meta_data.json', 'w') as f:
    json.dump(meta_data, f, indent=2)

files = [
        'K562_essential_normalized_singlecell_01.h5ad',
        'K562_essential_raw_singlecell_01.h5ad'
        ]

for file in files:
    url = None
    mfiles = meta_data['files']
    for f in mfiles:
        if f['name'] == file:
            url = f['download_url']

    if url is None:
        raise ValueError(f"File {file} not found in meta_data.json")

    print(f"Downloading {file} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('Content-Length', 0))
    with open(f'data/replogle/{file}', 'wb') as f, tqdm(
        desc=file,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved {file}")
