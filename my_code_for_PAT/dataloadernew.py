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
a=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine2_1_10800_70.mat','r')#,'r'
qwe = [a[element[0]][:] for element in a['djg']]

a2=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_1_10800_70.mat','r')
qwe2 = [a2[element[0]][:] for element in a2['djg']]

# a3=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_1_4413_70.mat','r')
# qwe3 = [a3[element[0]][:] for element in a3['djg']]

a4=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine34_1_6000_70.mat','r')
qwe4 = [a4[element[0]][:] for element in a4['djg']]

a5=h5py.File('/data1/MIP1/dataset/newmachine128/newmachinep1_1_7500_70.mat','r')
qwe5 = [a5[element[0]][:] for element in a5['djg']]
a7=h5py.File('/data1/MIP1/dataset/newmachine128/newmachinep2_1_4500_70.mat','r')
qwe7 = [a7[element[0]][:] for element in a7['djg']]
a57=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachinep3_1_3000_70.mat')['djg']
qwe57 = []
for i in range(3000):
    for j in range(1):
        qwe57.append(a57[j][i].T)

a6=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_30_360_70_rand6.mat','r')
db = []
qwe6 = []
for i in range(30):
    da = [a6[element[i]][:] for element in a6['djg']]
    db.append(da)
for i in range(360):
    for j in range(30):
        qwe6.append(db[j][i])

# a8=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_1_5400_70_around.mat','r')
# qwe8 = [a8[element[0]][:] for element in a8['djg']]

# a9=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_30_360_70_rand5.mat','r')
# db = []
# qwe9 = []
# for i in range(30):
#     da = [a9[element[i]][:] for element in a9['djg']]
#     db.append(da)
# for i in range(360):
#     for j in range(30):
#         qwe9.append(db[j][i])

qwe50 = qwe+qwe2+qwe4+qwe5+qwe7+qwe57+qwe6#+qwe8
line=[]
for i in range(129,385):
    line.append(i)
data=[]
for i in range(53400):
    aa = np.array(qwe50[i])
    # aa = aa[line,:]
    # aa += np.random.normal(0, 0.1, (256, 128))#addnoise
    # aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_NEAREST)  #
    data.append(zscorenorm(aa))

b=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine2_1_10800_true.mat')['djg3']
twe = []
for i in range(10800):
    for j in range(1):
        twe.append(b[j][i])

b2=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_1_10800_true.mat')['djg3']
twe2 = []
for i in range(10800):
    for j in range(1):
        twe2.append(b2[j][i])

# b3=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_1_4413_true.mat')['djg3']
# twe3 = []
# for i in range(4413):
#     for j in range(1):
#         twe3.append(b3[j][i])

b4=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine34_1_6000_true.mat')['djg3']
twe4 = []
for i in range(6000):
    for j in range(1):
        twe4.append(b4[j][i])

b5=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachinep1_1_7500_true.mat')['djg3']
twe5 = []
for i in range(7500):
    for j in range(1):
        twe5.append(b5[j][i])
b7=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachinep2_1_4500_true.mat')['djg3']
twe7 = []
for i in range(4500):
    for j in range(1):
        twe7.append(b7[j][i])
b57=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachinep3_1_3000_true.mat')['djg3']
twe57 = []
for i in range(3000):
    for j in range(1):
        twe57.append(b57[j][i])

b6=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_30_360_true_rand6.mat')['djg3']
twe6 = []
for i in range(360):
    for j in range(30):
        twe6.append(b6[j][i])

# b8=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_1_5400_true_around.mat')['djg3']
# twe8 = []
# for i in range(5400):
#     for j in range(1):
#         twe8.append(b8[j][i])

# b9=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_30_360_true_rand5.mat')['djg3']
# twe9 = []
# for i in range(360):
#     for j in range(30):
#         twe9.append(b9[j][i])

twe50 = twe + twe2 + twe4 + twe5 + twe7 + twe57 + twe6 #+ twe8
target=[]
for i in range(53400):
    bb=np.array(twe50[i].T)
    # bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
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
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform=transforms.Compose([
    # transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.ToTensor(),
    # normalize, #对tensor归一化需要放在ToTensor后面
])
transform_target=transforms.Compose([
    transforms.RandomRotation(degrees=(90,90)),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.ToTensor(),
    # normalize,
])
#验证数据对
# val_a=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine34_1_6000_70.mat','r')
# val_qwe = [val_a[element[0]][:] for element in val_a['djg']]

val_a=h5py.File('/data1/MIP1/dataset/newmachine128/newmachine_30_360_70_rand5.mat','r')  #,'r'
val_db = []
val_qwe = []
for i in range(30):
    val_da = [val_a[element[i]][:] for element in val_a['djg']]
    val_db.append(val_da)
for i in range(360):
    for j in range(30):
        val_qwe.append(val_db[j][i])

val_data=[]
for i in range(10800):
    aa = np.array(val_qwe[i])
    # aa = aa[line, :]
    # aa += np.random.normal(0, 0.1, (256, 128))
    # aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_NEAREST)  #
    val_data.append(zscorenorm(aa))

# val_b=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine34_1_6000_true.mat')['djg3']
# val_twe = []
# for i in range(6000):
#     for j in range(1):
#         val_twe.append(val_b[j][i])

val_b=scipy.io.loadmat('/data1/MIP1/dataset/newmachine128/newmachine_30_360_true_rand5.mat')['djg3']
val_twe = []
for i in range(360):
    for j in range(30):
        val_twe.append(val_b[j][i])

val_target=[]
for i in range(10800):
    bb=np.array(val_twe[i].T) #
    # bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    val_target.append(zscorenorm(bb))

#使用dataloader处理dataset
train_data=MyDataset(data,target,transform=None,transform_target=None)#type(train_data)= '__main__.MyDataset'
valid_data=MyDataset(val_data,val_target,transform=None,transform_target=None)
# train_data2=MyDataset(data,target,transform=transform,transform_target=transform_target)
# valid_data2=MyDataset(val_data,val_target,transform=transform,transform_target=transform_target)
BATCH_SIZE=32
train_loader=DataLoader(train_data,BATCH_SIZE,True)
valid_loader=DataLoader(valid_data,BATCH_SIZE,True)
# train_loader=DataLoader(train_data + train_data2,BATCH_SIZE,True)#type(train_loader)= torch.utils.data.dataloader.DataLoader
# valid_loader=DataLoader(valid_data + valid_data2,BATCH_SIZE,True)
