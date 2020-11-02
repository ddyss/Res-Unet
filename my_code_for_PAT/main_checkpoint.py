from __future__ import division
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
import numpy as np
import cv2
# '''
a=h5py.File('/data1/MIP1/djg/e2e/s_rand_30_360_70.mat')  #F:/colorbar/CNN/train/ F:/train/end to end 4560noiseball/
da = []
db = []
qwe = []
for i in range(30):
    da = [a[element[i]][:] for element in a['djg']]
    db.append(da)
for i in range(360):
    for j in range(30):
        qwe.append(db[j][i])
dys=[]
data=[]
for i in range(10800):
    dys=np.array(qwe[i])    #确实能起到转置作用，但在这转置无所谓，只要下面res形状就行
    res=cv2.resize(dys,(128,2560),interpolation=cv2.INTER_CUBIC)  #立方插值
    data.append(res)   #data和res横纵相反   2560*128

b=h5py.File('/data1/MIP1/djg/e2e/s_rand_30_360_true.mat')
ta = []
tb = []
twe = []
for i in range(30):
    ta = [b[element[i]][:] for element in b['djg2']]
    tb.append(ta)

for i in range(360):
    for j in range(30):
        twe.append(tb[j][i])

bys=[]
target=[]
for i in range(10800):
    bys=np.array(twe[i])
    bes=cv2.resize(bys,(128,128),interpolation=cv2.INTER_CUBIC)  #立方插值
    b2 = bes.copy()
    b2[b2 < 0] = 0
    b2[b2 > 0.3] = 3
    target.append(b2)

class MyDataset(Dataset):
    def __init__(self,data,target,transform):
        self.data=data
        self.target=target
        self.transform=transform
    def __getitem__(self,index):
        data=self.data[index]
        data=data.reshape(1,2560,128)
        data=data.astype(np.float32)
        target=self.target[index]
        target=target.astype(np.float32)
        target=target.reshape(1,128,128)
        return data,target
    def __len__(self):
        return len(self.data)
# '''
'''
a=scipy.io.loadmat('60_15_70.mat')['djg']
data = []
dwe=[]
for i in range(15):
    for j in range(60):
        dwe.append(a[j][i])
for i in range(900):
    dys=np.array(dwe[i])
    des=cv2.resize(dys,(128,2560),interpolation=cv2.INTER_CUBIC)
    data.append(des)

b=scipy.io.loadmat('60_15_true.mat')['djg2']
target=[]
twe=[]
for i in range(15):
    for j in range(60):
        twe.append(b[j][i])
for i in range(900):
    tys=np.array(twe[i])
    tes=cv2.resize(tys,(128,128),interpolation=cv2.INTER_CUBIC)
    target.append(tes)

class MyDataset(Dataset):
    def __init__(self,data,target,transform):
        self.data=data
        self.target=target
        self.transform=transform
    def __getitem__(self,index):
        data=self.data[index]
        data=data.reshape(1,2560,128)
        data=data.astype(np.float32)
        # data=data*1000
        target=self.target[index]
        target=target.astype(np.float32)
        target=target.reshape(1,128,128)
        # target=target*1000
        return data,target
    def __len__(self):
        return len(self.data)
# '''
# '''
a2 = scipy.io.loadmat('/data1/MIP1/djg/e2e/120_15_up_70.mat')['djg']
data2 = []
dwe2 = []
for i in range(15):
    for j in range(120):
        dwe2.append(a2[j][i])
for i in range(1800):
    dys2 = np.array(dwe2[i])
    des2 = cv2.resize(dys2, (128, 2560), interpolation=cv2.INTER_CUBIC)
    data2.append(des2)

b2 = scipy.io.loadmat('/data1/MIP1/djg/e2e/120_15_up_true.mat')['djg2']
target2 = []
twe2 = []
for i in range(15):
    for j in range(120):
        twe2.append(b2[j][i])
for i in range(1800):
    tys2 = np.array(twe2[i])
    tes2 = cv2.resize(tys2, (128, 128), interpolation=cv2.INTER_CUBIC)
    target2.append(tes2)
class MyDataset2(Dataset):
    def __init__(self,data2,target2,transform):
        self.data2=data2
        self.target2=target2
        self.transform=transform
    def __getitem__(self,index):
        data2=self.data2[index]
        data2 = data2.astype(np.float32)
        # data2 = self.transform(data2)
        data2 =data2.reshape(1,2560,128)
        target2=self.target2[index]
        target2=target2.astype(np.float32)
        # target2=self.transform(target2)
        target2=target2.reshape(1,128,128)
        return data2,target2
    def __len__(self):
        return len(self.data2)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   #会报tensor通道的错
])
dataset=MyDataset(data=data,target=target,transform=transform)
dataset2=MyDataset2(data2=data2,target2=target2,transform=transform)
# '''
# dd = torch.randn(4,1,128,512)
# dd = dd.cuda()
class DjgNet(nn.Module):
    def __init__(self):
        super(DjgNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)

        self.conv9 = nn.Conv2d(256, 256, [20, 3], [20, 1], 1)
        self.conv93 = nn.Conv2d(128, 128, [20, 3], [20, 1], 1)
        self.conv92 = nn.Conv2d(64, 64, [20, 3], [20, 1], 1)
        self.conv91 = nn.Conv2d(32, 32, [20, 3], [20, 1], 1)

        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        self.conv63 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv62 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv61 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.conv7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv71 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv72 = nn.Conv2d(32, 1, 3, 1, 1)

        self.conv8 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)#, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(x)
        x1r = self.relu(x)
        x = self.pool1(x1r)
        x = self.conv2(x)
        x = self.bn3(x)
        x2r = self.relu(x)
        x = self.pool1(x2r)
        x = self.conv3(x)
        x = self.bn4(x)
        x3r = self.relu(x)
        x = self.pool1(x3r)
        x = self.conv4(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.relu(x)  # 1,128,16,16

        xup3 = self.conv63(x)
        xadd1 = self.conv93(x3r) + xup3  # 1,128,32,32  简单的相加不会改变tensor的形状  cat则是连接，会改变
        x = self.conv7(xadd1)
        x = self.bn3(x)
        x = self.relu(x)  # 1,64,32,32

        xup2 = self.conv62(x)
        xadd2 = self.conv92(x2r) + xup2  # 1,64,64,64
        x = self.conv71(xadd2)
        x = self.bn2(x)
        x = self.relu(x)

        xup1 = self.conv61(x)
        xadd1 = self.conv91(x1r) + xup1  #
        x = self.conv72(xadd1)
        x = self.bn1(x)
        x = self.relu(x)

        return x

# djgnet = Net()
# # djgnet=nn.DataParallel(djgnet).to('cuda')
# djgnet=djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)  #
# '''
djgnet = DjgNet()
# djgnet=nn.DataParallel(djgnet).to('cuda')
djgnet=djgnet.to('cuda')
dataloader=DataLoader(dataset=dataset,batch_size=16,shuffle=True)
dataloader2=DataLoader(dataset=dataset2,batch_size=16,shuffle=True)
loss_func=nn.MSELoss()
optimizer=optim.Adam(djgnet.parameters(),lr=0.0001,weight_decay = 0.00001)
# optimizer = torch.optim.Adadelta(djgnet.parameters(), rho = 0.95, weight_decay = 0.001)
losses_his =[]
# losses_his2 =[]
for epoch in range(50):
    # train_loss = 0
    # valid_loss = 0
    for i,(data,target)in enumerate(dataloader):
        data=data.cuda()
        target=target.cuda()
        out=djgnet(data)
        loss=loss_func(out,target) #RuntimeError: The size of tensor a (8) must match the size of tensor b (200) at non-singleton dimension 3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_loss += loss.item()
        losses_his.append(loss.item())
        print(epoch,' ',i,':',loss.item())
    # with torch.no_grad():
    #     djgnet.eval()
    #     for i,(data2, target2) in enumerate(dataloader2):
    #         data2=data2.cuda()
    #         target2=target2.cuda()
    #         out2=djgnet(data2)
    #         loss2=loss_func(out2,target2)
    #         valid_loss += loss2.item()
    # train_loss = train_loss / len(dataloader.dataset)
    # valid_loss = valid_loss / len(dataloader2.dataset)
    # losses_his.append(train_loss)
    # losses_his2.append(valid_loss)
x= np.array(losses_his)
np.save('loss.npy',x)
# x2= np.array(losses_his2)
# np.save('loss2.npy',x2)
torch.save(djgnet,'unet_paper.pkl')

# plottrain=np.load('loss.npy')
# plt.plot(plottrain,'r-',lw=5)
# plotval=np.load('loss2.npy')
# plt.plot(plotval,'b-',lw=5)
# plt.title('r-train,b-val')
# plt.show()
# '''
# '''
c=scipy.io.loadmat('e90r31_70.mat')['sensor_data']  #F:/test/ball/ball.mat F:/trainpre/spider/several subset of  spider/
c = c.T
c = cv2.resize(c, (128, 2560), interpolation=cv2.INTER_CUBIC)
c=c.astype(np.float32)
c=torch.tensor(c)
c=c.reshape(1,1,2560,128)
c = c.type(torch.FloatTensor)
c = c.cuda()
with torch.no_grad():
    out=djgnet(c)
# out=out/1000
out=torch.Tensor.cpu(out)
out=np.asarray(out)
print(out.shape)
aaa=out.reshape(128,128)
scipy.io.savemat('pree90r31_70.mat',
                 {'testPredict': aaa})
# fig, (ax) = plt.subplots(nrows=1, figsize=(6,6))
# im = ax.imshow(aaa, extent=[0, 1, 0, 1])
# position=fig.add_axes([0.15, 0.05, 0.7, 0.03])#位置[左,下,右,上]
# cb=plt.colorbar(im,cax=position,orientation='horizontal')
# plt.show()
# '''

