from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import h5py
import numpy as np
import cv2
from PIL import Image
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
#load数据
a=h5py.File('/data1/MIP1/dataset/dataset_shuttle.h5','r')#  mixed
qwe_ori = a['data'][:]
twe_ori = a['target'][:]

# line=[]
# for i in range(129,385):
#     line.append(i)
qwe50 = qwe_ori[0:58126]
data=[]
for i in range(58126):
    aa_ori = np.array(qwe50[i])
    # aa_ori = aa_ori[line,:]
    # aa_ori=cv2.resize(aa_ori,(128,128),interpolation=cv2.INTER_NEAREST)  #
    aa_ori = zscorenorm(aa_ori)
    data.append(aa_ori)
print(len(data))
target=[]
twe50 = twe_ori[0:58126]
for i in range(58126):
    bb=np.array(twe50[i].T)
    bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bb = zscorenorm(bb)
    target.append(bb)

val_qwe50 = qwe_ori[58126:72658]
val_data=[]
for i in range(14532):
    aa_ori = np.array(val_qwe50[i])
    # aa_ori = aa_ori[line, :]
    # aa_ori=cv2.resize(aa_ori,(128,128),interpolation=cv2.INTER_NEAREST)  #
    aa_ori = zscorenorm(aa_ori)
    val_data.append(aa_ori)

val_twe50 = twe_ori[58126:72658]
val_target=[]
for i in range(14532):
    bb=np.array(val_twe50[i].T) #
    bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bb = zscorenorm(bb)
    val_target.append(bb)

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
    # transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.ToTensor(),
])
transform_target=transforms.Compose([
    transforms.RandomRotation(degrees=(90,90)),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.ToTensor()
])

#使用dataloader处理dataset
train_data=MyDataset(data,target,transform=None,transform_target=None)
valid_data=MyDataset(val_data,val_target,transform=None,transform_target=None)
# train_data2=MyDataset(data,target,transform=transform,transform_target=transform_target)
# valid_data2=MyDataset(val_data,val_target,transform=transform,transform_target=transform_target)
BATCH_SIZE=32
train_loader=DataLoader(train_data,BATCH_SIZE,True)
valid_loader=DataLoader(valid_data,BATCH_SIZE,True)
# train_loader=DataLoader(train_data + train_data2,BATCH_SIZE,True)
# valid_loader=DataLoader(valid_data + valid_data2,BATCH_SIZE,True)

