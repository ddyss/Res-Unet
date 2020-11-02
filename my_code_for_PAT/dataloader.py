import scipy.io
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
#训练数据对
a=h5py.File('../newmachine2_1_10800_70.mat','r')#,'r'
qwe_ori = [a[element[0]][:] for element in a['djg']]
qwe = qwe_ori[0:5400]
print(len(qwe))
twe10 = qwe_ori[5400:10800]
print(len(twe10))

a2=h5py.File('../newmachine_1_10800_70.mat','r')
qwe2 = [a2[element[0]][:] for element in a2['djg']]
qwe2 = qwe2[0:5400]
twe20 = qwe2[5400:10800]

a3=scipy.io.loadmat('../newmachine_1_3077_70.mat')['djg']
qwe3 = []
for i in range(3077):
    for j in range(1):
        qwe3.append(a3[j][i].T)
del qwe3[1486]
del qwe3[492]
del qwe3[454]

a4=h5py.File('../newmachine34_1_6000_70.mat','r')
qwe4 = [a4[element[0]][:] for element in a4['djg']]
# qwe4 = qwe4[0:1000]

a5=h5py.File('../newmachine_1_15000_70.mat','r')
qwe5 = [a5[element[0]][:] for element in a5['djg']]
# qwe5 = qwe5[0:3000]

a6=h5py.File('../newmachine_30_360_70_rand6.mat','r')
db = []
qwe6 = []
for i in range(30):
    da = [a6[element[i]][:] for element in a6['djg']]
    db.append(da)
for i in range(360):
    for j in range(30):
        qwe6.append(db[j][i])
# qwe6 = qwe6[0:1000]

a7=h5py.File('../newmachine_1_5400_70.mat','r')
qwe7 = [a7[element[0]][:] for element in a7['djg']]
# qwe7 = qwe7[0:1000]

qwe50 = qwe+qwe2+qwe3+qwe4+qwe5+qwe6+qwe7
line=[]
for i in range(129,385):
    line.append(i)
data=[]
# data_f = []
import scipy.signal as signal
def stft_pic(img):
    for i in range(128):
        begin = np.zeros(shape=(257,128*3))
        aaa = signal.stft(img[:,i],nperseg=512,nfft=512)
        begin[:,i:i+3] = aaa[2]
        begin = cv2.resize(begin,(128,512),interpolation=cv2.INTER_NEAREST)
    return begin
for i in range(61874):
    aa_ori = np.array(qwe50[i])
    # aa = aa[line,:]
    # aa += np.random.normal(0, 0.1, (256, 128))#addnoise
    # aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_NEAREST)  #
    # aa = stft_pic(aa_ori)
    # new = zscorenorm(aa)
    # data_f.append(new)

    new = zscorenorm(aa_ori)
    data.append(new)

b=scipy.io.loadmat('../newmachine2_1_10800_true.mat')['djg3']
twe = []
for i in range(10800):
    for j in range(1):
        twe.append(b[j][i])
# twe = twe[0:1000]

b2=scipy.io.loadmat('../newmachine_1_10800_true.mat')['djg3']
twe2 = []
for i in range(10800):
    for j in range(1):
        twe2.append(b2[j][i])
# twe2 = twe2[0:1000]

b3=scipy.io.loadmat('../newmachine_1_3077_true.mat')['djg3']
twe3 = []
for i in range(3077):
    for j in range(1):
        twe3.append(b3[j][i])
del twe3[1486]
del twe3[492]
del twe3[454]

b4=scipy.io.loadmat('../newmachine34_1_6000_true.mat')['djg3']
twe4 = []
for i in range(6000):
    for j in range(1):
        twe4.append(b4[j][i])
# twe4 = twe4[0:1000]

b5=scipy.io.loadmat('../newmachine_1_15000_true.mat')['djg3']
twe5 = []
for i in range(15000):
    for j in range(1):
        twe5.append(b5[j][i])
# twe5 = twe5[0:3000]

b6=scipy.io.loadmat('../newmachine_30_360_true_rand6.mat')['djg3']
twe6 = []
for i in range(360):
    for j in range(30):
        twe6.append(b6[j][i])
# twe6 = twe6[0:1000]

b7=scipy.io.loadmat('../newmachine_1_5400_true.mat')['djg3']
twe7 = []
for i in range(5400):
    for j in range(1):
        twe7.append(b7[j][i])
# twe7 = twe7[0:1000]

twe50 = twe + twe2 + twe3 + twe4 + twe5 + twe6 + twe7
target=[]
for i in range(61874):
    bb=np.array(twe50[i].T)
    bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bbnew = zscorenorm(bb)
    target.append(bbnew)

class MyDataset(Dataset):
    def __init__(self,data,target,transform=None,transform_target=None):
        self.data=data
        # self.data_f = data_f
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

        # data_f = self.data_f[index]
        # data_f = data_f.astype(np.float32)
        # data_f = Image.fromarray(data_f)
        # if self.transform is not None:
        #     data_f = self.transform(data_f)
        # data_f = np.array(data_f)
        # data_f = data_f.reshape(1, 512, 128)

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
#验证数据对
val_a=h5py.File('../newmachine_30_360_70_rand5.mat','r')  #,'r'
val_db = []
val_qwe = []
for i in range(30):
    val_da = [val_a[element[i]][:] for element in val_a['djg']]
    val_db.append(val_da)
for i in range(360):
    for j in range(30):
        val_qwe.append(val_db[j][i])

val_data=[]
# val_data_f = []
for i in range(10800):
    aa_ori = np.array(val_qwe[i])
    # aa = aa[line, :]
    # aa += np.random.normal(0, 0.1, (256, 128))
    # aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_NEAREST)  #
    # aa = stft_pic(aa_ori)
    # new = zscorenorm(aa)
    # val_data_f.append(new)

    new = zscorenorm(aa_ori)
    val_data.append(new)

val_b=scipy.io.loadmat('../newmachine_30_360_true_rand5.mat')['djg3']
val_twe = []
for i in range(360):
    for j in range(30):
        val_twe.append(val_b[j][i])

val_target=[]
for i in range(10800):
    bb=np.array(val_twe[i].T) #
    bb=cv2.resize(bb,(128,128),interpolation=cv2.INTER_NEAREST)  #
    bbnew = zscorenorm(bb)
    val_target.append(bbnew)

#使用dataloader处理dataset
train_data=MyDataset(data,target,transform=None,transform_target=None)
valid_data=MyDataset(val_data,val_target,transform=None,transform_target=None)
# train_data2=MyDataset(data,target,transform=transform,transform_target=transform_target)
# valid_data2=MyDataset(val_data,val_target,transform=transform,transform_target=transform_target)
BATCH_SIZE=16
train_loader=DataLoader(train_data,BATCH_SIZE,True)
valid_loader=DataLoader(valid_data,BATCH_SIZE,True)
# train_loader=DataLoader(train_data + train_data2,BATCH_SIZE,True)
# valid_loader=DataLoader(valid_data + valid_data2,BATCH_SIZE,True)
