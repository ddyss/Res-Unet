import scipy.io
import h5py
# '''
# load数据
raw_a=h5py.File('../newmachine2_1_10800_70.mat','r')
raw_qwe = [raw_a[element[0]][:] for element in raw_a['djg']]
# for i in range(10800):
#     for j in range(1):
#         raw_qwe.append(raw_a[j][i].T)

raw_a2=h5py.File('../newmachine_1_10800_70.mat','r')
raw_qwe2 = [raw_a2[element[0]][:] for element in raw_a2['djg']]
# for i in range(10800):
#     for j in range(1):
#         raw_qwe2.append(raw_a2[j][i].T)

raw_a3=scipy.io.loadmat('../newmachine_1_3077_70.mat')['djg']
raw_qwe3 = []
for i in range(3077):
    for j in range(1):
        raw_qwe3.append(raw_a3[j][i].T)
del raw_qwe3[2559-1]
del raw_qwe3[1505-1]
del raw_qwe3[1502-1]
del raw_qwe3[1492-1]
del raw_qwe3[1491-1]
del raw_qwe3[1487-1]
del raw_qwe3[1485-1]
del raw_qwe3[1483-1]
del raw_qwe3[1482-1]
del raw_qwe3[1481-1]
del raw_qwe3[1480-1]
del raw_qwe3[1476-1]
del raw_qwe3[544-1]
del raw_qwe3[525-1]
del raw_qwe3[524-1]
del raw_qwe3[493-1]
del raw_qwe3[481-1]
del raw_qwe3[455-1]
del raw_qwe3[454-1]

raw_a4=h5py.File('../newmachine34_1_6000_70.mat','r')
raw_qwe4 = [raw_a4[element[0]][:] for element in raw_a4['djg']]
# for i in range(6000):
#     for j in range(1):
#         raw_qwe4.append(raw_a4[j][i].T)

raw_a5=h5py.File('../newmachine_1_15000_70.mat','r')
raw_qwe5 = [raw_a5[element[0]][:] for element in raw_a5['djg']]
# for i in range(15000):
#     for j in range(1):
#         raw_qwe5.append(raw_a5[j][i].T)

raw_a6=h5py.File('../newmachine_30_360_70_rand6.mat','r')
raw_db6 = []
raw_qwe6 = []
for i in range(30):
    raw_da6 = [raw_a6[element[i]][:] for element in raw_a6['djg']]
    raw_db6.append(raw_da6)
for i in range(360):
    for j in range(30):
        raw_qwe6.append(raw_db6[j][i])

raw_a7=h5py.File('../newmachine_1_5400_70.mat','r')
raw_qwe7 = [raw_a7[element[0]][:] for element in raw_a7['djg']]
# for i in range(5400):
#     for j in range(1):
#         raw_qwe7.append(raw_a7[j][i].T)

raw_a8=h5py.File('../newmachine_30_360_70_rand5.mat','r')
raw_db8 = []
raw_qwe8 = []
for i in range(30):
    raw_da8 = [raw_a8[element[i]][:] for element in raw_a8['djg']]
    raw_db8.append(raw_da8)
for i in range(360):
    for j in range(30):
        raw_qwe8.append(raw_db8[j][i])


a=scipy.io.loadmat('../newmachine2_1_10800_70_bp_recon.mat')['djg_bp_recon']
qwe = []
for i in range(10800):
    for j in range(1):
        qwe.append(a[j][i])

a2=scipy.io.loadmat('../newmachine_1_10800_70_bp_recon.mat')['djg_bp_recon']
qwe2 = []
for i in range(10800):
    for j in range(1):
        qwe2.append(a2[j][i])

a3=scipy.io.loadmat('../newmachine_1_3077_70_bp_recon.mat')['djg_bp_recon']
qwe3 = []
for i in range(3077):
    for j in range(1):
        qwe3.append(a3[j][i])
del qwe3[2559-1]
del qwe3[1505-1]
del qwe3[1502-1]
del qwe3[1492-1]
del qwe3[1491-1]
del qwe3[1487-1]
del qwe3[1485-1]
del qwe3[1483-1]
del qwe3[1482-1]
del qwe3[1481-1]
del qwe3[1480-1]
del qwe3[1476-1]
del qwe3[544-1]
del qwe3[525-1]
del qwe3[524-1]
del qwe3[493-1]
del qwe3[481-1]
del qwe3[455-1]
del qwe3[454-1]

a4=scipy.io.loadmat('../newmachine34_1_6000_70_bp_recon.mat')['djg_bp_recon']
qwe4 = []
for i in range(6000):
    for j in range(1):
        qwe4.append(a4[j][i])

a5=scipy.io.loadmat('../newmachine_1_15000_70_bp_recon.mat')['djg_bp_recon']
qwe5 = []
for i in range(15000):
    for j in range(1):
        qwe5.append(a5[j][i])

a6=scipy.io.loadmat('../newmachine_30_360_70_rand6_bp_recon.mat')['djg_bp_recon']
qwe6 = []
for i in range(360):
    for j in range(30):
        qwe6.append(a6[j][i])

a7=scipy.io.loadmat('../newmachine_1_5400_70_bp_recon.mat')['djg_bp_recon']
qwe7 = []
for i in range(5400):
    for j in range(1):
        qwe7.append(a7[j][i])

a8=scipy.io.loadmat('../newmachine_30_360_70_rand5_bp_recon.mat')['djg_bp_recon']
qwe8 = []
for i in range(360):
    for j in range(30):
        qwe8.append(a8[j][i])


b=scipy.io.loadmat('../newmachine2_1_10800_true.mat')['djg3']
twe = []
for i in range(10800):
    for j in range(1):
        twe.append(b[j][i])

b2=scipy.io.loadmat('../newmachine_1_10800_true.mat')['djg3']
twe2 = []
for i in range(10800):
    for j in range(1):
        twe2.append(b2[j][i])

b3=scipy.io.loadmat('../newmachine_1_3077_true.mat')['djg3']
twe3 = []
for i in range(3077):
    for j in range(1):
        twe3.append(b3[j][i])
del twe3[2559-1]
del twe3[1505-1]
del twe3[1502-1]
del twe3[1492-1]
del twe3[1491-1]
del twe3[1487-1]
del twe3[1485-1]
del twe3[1483-1]
del twe3[1482-1]
del twe3[1481-1]
del twe3[1480-1]
del twe3[1476-1]
del twe3[544-1]
del twe3[525-1]
del twe3[524-1]
del twe3[493-1]
del twe3[481-1]
del twe3[455-1]
del twe3[454-1]

b4=scipy.io.loadmat('../newmachine34_1_6000_true.mat')['djg3']
twe4 = []
for i in range(6000):
    for j in range(1):
        twe4.append(b4[j][i])

b5=scipy.io.loadmat('../newmachine_1_15000_true.mat')['djg3']
twe5 = []
for i in range(15000):
    for j in range(1):
        twe5.append(b5[j][i])

b6=scipy.io.loadmat('../newmachine_30_360_true_rand6.mat')['djg3']
twe6 = []
for i in range(360):
    for j in range(30):
        twe6.append(b6[j][i])

b7=scipy.io.loadmat('../newmachine_1_5400_true.mat')['djg3']
twe7 = []
for i in range(5400):
    for j in range(1):
        twe7.append(b7[j][i])

b8=scipy.io.loadmat('../newmachine_30_360_true_rand5.mat')['djg3']
twe8 = []
for i in range(360):
    for j in range(30):
        twe8.append(b8[j][i])
# '''

import random
raw_data_ori = raw_qwe + raw_qwe2 + raw_qwe3 + raw_qwe4 + raw_qwe5 + raw_qwe6 + raw_qwe7 + raw_qwe8
data_ori = qwe + qwe2 + qwe3 + qwe4 + qwe5 + qwe6 + qwe7 + qwe8
target_ori = twe + twe2 + twe3 + twe4 + twe5 + twe6 + twe7 + twe8
print(len(raw_data_ori))
# 打乱1次
randnum = random.randint(0,10000)
random.seed(randnum)
random.shuffle(raw_data_ori)
random.seed(randnum)
random.shuffle(data_ori)
random.seed(randnum)
random.shuffle(target_ori)
# 再打乱1次
cc = list(zip(raw_data_ori,data_ori, target_ori))
random.shuffle(cc)
raw_data_ori[:], data_ori[:], target_ori[:] = zip(*cc)

f = h5py.File('../dataset_ynet_shuttle.h5', 'w')
f.create_dataset('data_raw', data=raw_data_ori)
f.create_dataset('data_bp', data=data_ori)
f.create_dataset('target', data=target_ori)
f.close()

raw_data_debug = raw_data_ori[0:10000]
data_debug = data_ori[0:10000]
target_debug = target_ori[0:10000]
f = h5py.File('../dataset_ynet_debug.h5', 'w')
f.create_dataset('data_raw', data=raw_data_debug)
f.create_dataset('data_bp', data=data_debug)
f.create_dataset('target', data=target_debug)
f.close()


