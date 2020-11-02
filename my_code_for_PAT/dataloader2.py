import scipy.io
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import h5py
import numpy as np
import cv2
#训练数据对
a=scipy.io.loadmat('../1_3077_70.mat')['djg']  #,'r'
qwe = []
for i in range(3077):
    for j in range(1):
        qwe.append(a[j][i].T)

qwe50 = qwe#+qwe2+qwe3+qwe4#+qwe6
data=[]
for i in range(3077):
    aa = np.array(qwe50[i])
    aa=cv2.resize(aa,(128,2048),interpolation=cv2.INTER_NEAREST)  #
    bb = np.mean(aa)
    cc = aa.std()
    new = (aa - bb) / (cc)
    data.append(new)

c=scipy.io.loadmat('../1_3077_100.mat')['djg']
qqwe = []
for i in range(3077):
    for j in range(1):
        qqwe.append(c[j][i].T) #

qqwe50 = qqwe#+ twe2 + twe3 + twe4 #+ twe5
qqtarget=[]
for i in range(3077):
    bys=np.array(qqwe50[i]) #
    bb=cv2.resize(bys,(128,2048),interpolation=cv2.INTER_NEAREST)  #
    bbmean = np.mean(bb)
    bbstd = bb.std()
    bbnew = (bb - bbmean) / (bbstd)
    # bb[bb<0] = 0
    qqtarget.append(bbnew)

b=h5py.File('../1_3077_true.mat')
twe = [b[element[0]][:] for element in b['djg2']]

twe50 = twe#+ twe2 + twe3 + twe4 #+ twe5
target=[]
for i in range(3077):
    bys=np.array(twe50[i]) #
    bb=cv2.resize(bys,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bbmean = np.mean(bb)
    bbstd = bb.std()
    bbnew = (bb - bbmean) / (bbstd)
    # bb[bb<0] = 0
    target.append(bbnew)

class MyDataset(Dataset):
    def __init__(self,data,qqtarget,target,transform):
        self.data=data
        self.qqtarget = qqtarget
        self.target=target
        self.transform=transform
    def __getitem__(self,index):
        data=self.data[index]
        data=data.reshape(1,2048,128)
        data=data.astype(np.float32)

        qqtarget = self.qqtarget[index]
        qqtarget = qqtarget.reshape(1, 2048, 128)
        qqtarget = qqtarget.astype(np.float32)

        target=self.target[index]
        target=target.astype(np.float32)
        target=target.reshape(1,128,128)
        return data,qqtarget,target
    def __len__(self):
        return len(self.data)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

'''
#验证数据对
val_a=h5py.File('../newmachine_30_360_70_rand5.mat')  #,'r'
val_da = []
val_db = []
val_qwe = []
for i in range(30):
    val_da = [val_a[element[i]][:] for element in val_a['djg']]
    val_db.append(val_da)
for i in range(360):
    for j in range(30):
        val_qwe.append(val_db[j][i])

val_data=[]
for i in range(10800//2):
    aa = np.array(val_qwe[i])
    aa=cv2.resize(aa,(128,2048),interpolation=cv2.INTER_NEAREST)  #
    bb = np.mean(aa)
    cc = aa.std()
    new = (aa - bb) / (cc)
    val_data.append(new)

val_b=scipy.io.loadmat('../newmachine_30_360_true_rand5.mat')['djg3']
val_twe = []
for i in range(360):
    for j in range(30):
        val_twe.append(val_b[j][i])

val_target=[]
for i in range(10800//2):
    val_bys=np.array(val_twe[i].T) #
    bb=cv2.resize(val_bys,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bbmean = np.mean(bb)
    bbstd = bb.std()
    bbnew = (bb - bbmean) / (bbstd)
    # bb[bb<0] = 0
    val_target.append(bbnew)
# '''

#使用dataloader处理dataset
train_data=MyDataset(data,qqtarget,target,transform=transform)
# valid_data=MyDataset(val_data,val_target,transform=transform)
BATCH_SIZE=4
train_loader=DataLoader(train_data,BATCH_SIZE,True)
# valid_loader=DataLoader(valid_data,BATCH_SIZE,True)

