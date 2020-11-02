import scipy.io
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import h5py
import numpy as np
import cv2
from PIL import Image
def max_min(x):
    aa = np.max(x)
    bb = np.min(x)
    new = (x-bb)/(aa-bb)
    return new
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
#训练数据对
a=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_119djg_scan2_data.mat')['djg3']
qwe = []
for i in range(276):
    for j in range(1):
        qwe.append(a[j][i])

a2=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_cuni_scan1_data.mat')['djg3']
qwe2 = []
for i in range(31):
    for j in range(1):
        qwe2.append(a2[j][i])

a3=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_cuso4_scan1_data.mat')['djg3']
qwe3 = []
for i in range(31):
    for j in range(1):
        qwe3.append(a3[j][i])

a4=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_irf_scan1_data.mat')['djg3']
qwe4 = []
for i in range(17):
    for j in range(1):
        qwe4.append(a4[j][i])

# a5=h5py.File('/data1/MIP1/dataset/realtrain/real_MOUSE_scan1_data.mat','r')
# qwe5 = [a5[element[0]][:] for element in a5['djg3']]
# a7=h5py.File('/data1/MIP1/dataset/realtrain/newmachinep2_1_4500_70.mat','r')
# qwe7 = [a7[element[0]][:] for element in a7['djg3']]

# a6=h5py.File('/data1/MIP1/dataset/realtrain/newmachine_30_360_70_rand6.mat','r')
# db = []
# qwe6 = []
# for i in range(30):
#     da = [a6[element[i]][:] for element in a6['djg3']]
#     db.append(da)
# for i in range(360):
#     for j in range(30):
#         qwe6.append(db[j][i])

# data = qwe+qwe2+qwe3+qwe4+qwe5+qwe6+qwe7
qwe50 = qwe+qwe2+qwe3+qwe4#+qwe5+qwe7+qwe6
# line=[]
# for i in range(129,385):
#     line.append(i)
data=[]
for i in range(355):
    aa = np.array(qwe50[i])
    # aa = aa[line,:]
    # aa += np.random.normal(0, 0.1, (256, 128))#addnoise
    aa=cv2.resize(aa,(128,512),interpolation=cv2.INTER_NEAREST)  #
    data.append(zscorenorm(aa))

b=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_119djg_scan2_recon.mat')['djg3']
twe = []
for i in range(276):
    for j in range(1):
        twe.append(b[j][i])

b2=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_cuni_scan1_recon.mat')['djg3']
twe2 = []
for i in range(31):
    for j in range(1):
        twe2.append(b2[j][i])

b3=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_cuso4_scan1_recon.mat')['djg3']
twe3 = []
for i in range(31):
    for j in range(1):
        twe3.append(b3[j][i])

b4=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/real_irf_scan1_recon.mat')['djg3']
twe4 = []
for i in range(17):
    for j in range(1):
        twe4.append(b4[j][i])

# b5=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/newmachinep1_1_7500_true.mat')['djg3']
# twe5 = []
# for i in range(7500):
#     for j in range(1):
#         twe5.append(b5[j][i])
# b7=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/newmachinep2_1_4500_true.mat')['djg3']
# twe7 = []
# for i in range(4500):
#     for j in range(1):
#         twe7.append(b7[j][i])

# b6=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/newmachine_30_360_true_rand6.mat')['djg3']
# twe6 = []
# for i in range(360):
#     for j in range(30):
#         twe6.append(b6[j][i])

# target = twe + twe2 + twe3 + twe4 + twe5 + twe6 + twe7
twe50 = twe + twe2 + twe3 + twe4# + twe5 + twe7 + twe6
target=[]
for i in range(355):
    bb=np.array(twe50[i])
    bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    target.append(zscorenorm(bb))

class MyDataset(Dataset):
    def __init__(self,data,target,transform=None,transform_target=None):
        self.data=data
        self.target=target
        self.transform=transform
        self.transform_target = transform_target
    def __getitem__(self,index):
        data=self.data[index]
        data=data.astype(np.float32)
        data=Image.fromarray(data)
        if self.transform is not None:
            data=self.transform(data)
        data=np.array(data)
        data=data.reshape(1,512,128)

        target=self.target[index]
        target=target.astype(np.float32)
        target=Image.fromarray(target)
        if self.transform_target is not None:
            target=self.transform_target(target)
        target=np.array(target)
        target=target.reshape(1,128,128)

        return data,target
    def __len__(self):
        return len(self.data)

transform=transforms.Compose([
    transforms.RandomVerticalFlip(p=1)
    # transforms.ToTensor()
])
transform_target=transforms.Compose([
    transforms.RandomRotation(degrees=(90,90)),
    transforms.RandomHorizontalFlip(p=1)
    # transforms.ToTensor()
])
#验证数据对
# val_a=h5py.File('/data1/MIP1/dataset/realtrain/newmachine_30_360_70_rand5.mat','r')  #,'r'
# val_a=h5py.File('/data1/MIP1/dataset/realtrain/newmachinep2_1_4500_70.mat','r')
# val_db = []
# val_qwe = []
# for i in range(30):
#     val_da = [val_a[element[i]][:] for element in val_a['djg3']]
#     val_db.append(val_da)
# for i in range(360):
#     for j in range(30):
#         val_qwe.append(val_db[j][i])
# val_qwe = [val_a[element[0]][:] for element in val_a['djg3']]

# val_data=[]
# for i in range(1000):
#     aa = np.array(val_qwe[i])
    # aa = aa[line, :]
    # aa += np.random.normal(0, 0.1, (256, 128))
    # aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_NEAREST)  #
    # val_data.append(zscorenorm(aa))

# val_b=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/newmachine_30_360_true_rand5.mat')['djg3']
# val_b=scipy.io.loadmat('/data1/MIP1/dataset/realtrain/newmachinep2_1_4500_true.mat')['djg3']
# val_twe = []
# for i in range(1000):
#     for j in range(1):
#         val_twe.append(val_b[j][i])


# val_target=[]
# for i in range(1000):
#     bb=np.array(val_twe[i].T) #
#     # bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
#     val_target.append(zscorenorm(bb))

#使用dataloader处理dataset
train_data=MyDataset(data,target,transform=None,transform_target=None)
# valid_data=MyDataset(val_data,val_target,transform=None,transform_target=None)
train_data2=MyDataset(data,target,transform=transform,transform_target=transform_target)
# valid_data2=MyDataset(val_data,val_target,transform=transform,transform_target=transform_target)
BATCH_SIZE=32
# train_loader=DataLoader(train_data,BATCH_SIZE,True)
# valid_loader=DataLoader(valid_data,BATCH_SIZE,True)
train_loader=DataLoader(train_data + train_data2,BATCH_SIZE,True)
# valid_loader=DataLoader(valid_data + valid_data2,BATCH_SIZE,True)
print(type(train_data))
print(type(train_data2))
print(type(train_loader))