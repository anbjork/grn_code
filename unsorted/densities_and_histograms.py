

def check_densities():

    print()
    print(f'{len(aa) = }')
    data = aa
    methods = [elem['meta']['method'] for elem in data]
    print()
    print(f'{sorted(set(methods)) = }')

    m = 'bigsm_T'
    # m = 'bigsm'

    print()
    print(f'{m = }')
    ms = [elem for elem in data if elem['meta']['method'] == m]
    print()
    print(f'{len(ms) = }')
    print()
    print(f'{ms[0] = }')
    # net = ms[0]['estimated_network']
    # net
    # net == 0
    # np.sum(net == 0)
    # import numppy as np
    # np.sum(net == 0)
    # (net == 0).sum().sum() / net.size
    # net.shape
    # net.size
    # nets = [elem['estimated_network'] for m in ms]
    nets = [elem['estimated_network'] for elem in ms]
    print()
    print(f'{len(nets) = }')
    nets[0]
    densities = [(net != 0).sum().sum() / net.size for net in nets]
    print()
    print(f'{densities = }')

    print()
    print(f'{min(densities) = }')





import anton_util
import numpy as np
import matplotlib.pyplot as plt

infile = 'inferences.pkl'
aa = anton_util.unpickle_object(infile)
data = aa

# check_densities()

methods = [elem['meta']['method'] for elem in data]

m = 'bigsm_T'
ms = [elem for elem in data if elem['meta']['method'] == m]
nets = [elem['estimated_network'] for elem in ms]

from pathlib import Path
outdir = Path('bigsm_histograms')
outdir.mkdir(exist_ok=True)

anton_util.log_timestamp('before making histograms')
for i, net in enumerate(nets):
    net = np.array(net)

    counts, bins = np.histogram(net.flatten(), bins=30)
    c2 = np.log10(counts + 1)

    plt.figure()
    plt.stairs(c2, bins, fill=True)
    plt.xlabel('value')
    plt.ylabel('log10(count + 1)')
    plt.grid(True)
    plt.savefig(outdir / f'{m}_sample_{i}_histogram.png')
    plt.close()

anton_util.log_timestamp('after making histograms')









