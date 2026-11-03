
import anton_util

infile = 'inferences.pkl'
aa = anton_util.unpickle_object(infile)
print()
print(f'{len(aa) = }')
data = aa
methods = [elem['meta']['method'] for elem in data]
print()
print(f'{sorted(set(methods)) = }')
m = 'bigsm_T'
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
densities = [(net == 0).sum().sum() / net.size for net in nets]
print()
print(f'{densities = }')




