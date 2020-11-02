import scipy.io
import numpy as np

# b3=scipy.io.loadmat('D:/newmachine128/newmachine_1_4649_true.mat')['djg3']
b3=scipy.io.loadmat('G:/newmachine_1_4413_true.mat')['djg3']
twe3 = []
for i in range(4413):
    for j in range(1):
        twe3.append(b3[j][i])

for i in range(4413):
    # aa = np.max(twe3[i]) - np.min(twe3[i])
    bb = np.std(twe3[i])
    if bb == 0:
        print('wrong num is',i)

