from dataloader_ynet_final import train_loader,valid_loader
import torch.nn as nn
from torch import optim
import torch
from timeit import default_timer as timer
import numpy as np
# from apex import amp#在调用amp.initialize之前，模型不能调用任何分布式设置函数。amp和两块GPU只能选其一
# import math
# from scipy.stats import pearsonr
torch.backends.cudnn.benchmark=True
import os,random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def seed_torch(seed=1001):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
# seed_torch()
from model_ynet import djgnet
net = djgnet.to('cuda')
net = nn.DataParallel(net)
model_name='unet_paper.pkl'
loss_func=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=0.0001,weight_decay=0.00001)#去掉decay会减弱
# net,optimizer=amp.initialize(net,optimizer,opt_level='O1',verbosity=0)
trainloss_total=[]
validloss_total=[]
# pc_iters = []
# psnr_iters = []
bestloss = np.Inf

# from manager_torch import GPUManager
# gm=GPUManager()
# with gm.auto_choice():
for epoch in range(50):
    start1=timer()
    trainloss=0
    validloss=0
    net.train()
    for i,(data,data2,target)in enumerate(train_loader):
        data=data.cuda()
        data2 = data2.cuda()
        target=target.cuda()
        out=net(data,data2)
        loss=loss_func(out,target)
        # loss2 = loss_func(data2, target)
        # loss = loss1 + 0.5 * loss2
        optimizer.zero_grad()
        # with amp.scale_loss(loss,optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        progress=round(100*(i+1)/len(train_loader),2)
        time = round(timer()-start1,2)
        show = round(loss.item(),6)
        # print(epoch, ' ', i, ':', loss.item(), end='\n')
        print(epoch,i, ':', show,' ',progress,'% complete','  ',time,'second passed',end='\r')
        # PSNR = psnr(out,target)
        # PC = pearsonr(out,target)[0]
        # pc_iters.append(PC)
        # psnr_iters.append(PSNR)

    with torch.no_grad():
        net.eval()
        start2 = timer()
        for i,(data,data2,target) in enumerate(valid_loader):
            data=data.cuda()
            data2 = data2.cuda()
            target=target.cuda()
            out=net(data,data2)
            loss=loss_func(out,target)
            validloss+=loss.item()
            progress = round(100 * (i + 1) / len(valid_loader), 2)
            time = round(timer() - start2, 2) #对浮点数进行4舍5入取值，但不总是，保留几位小数,不写的话默认保留到整数。
            show = round(loss.item(),6)
            # print(epoch, ' ', i, ':', loss.item(), end='\n')
            print(epoch,i, ':', show,' ',progress, '% complete','  ', time, 'second passed', end='\r')
    trainloss=trainloss/len(train_loader.dataset)
    validloss=validloss/len(valid_loader.dataset)

    trainloss_total.append(trainloss)
    np.save('trainloss.npy', trainloss_total)
    validloss_total.append(validloss)
    np.save('validloss.npy', validloss_total)
    torch.save(net, '/data1/MIP1/dataset/segnet/debug/unet_paper' + str(epoch) + '.pkl')
    np.save('/data1/MIP1/dataset/segnet/debug/trainloss.npy', trainloss_total)
    np.save('/data1/MIP1/dataset/segnet/debug/validloss.npy', validloss_total)
    # np.save('/data1/MIP1/dataset/segnet/debug/pc_iters.npy', pc_iters)
    # np.save('/data1/MIP1/dataset/segnet/debug/psnr_iters.npy', psnr_iters)

    if validloss<bestloss:
        torch.save(net,model_name)
        bestloss=validloss
        bestepoch=epoch

    trainloss = round(trainloss, 6)
    validloss = round(validloss, 6)
    print('epoch:',epoch,'trainloss:',trainloss,'validloss:',validloss)

bestloss = round(bestloss,6)
print('best epoch:',bestepoch,'  ','bestloss:',bestloss)

'''
roottest = '/data1/MIP1/dataset/segnet/'
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb) / cc
    return aaa
def psnr(img1, img2):
    diff = img1 - img2
    diff = diff.flatten('C')#这样对图片也适用
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
def testpre(c):
    c = c.T
    c = zscorenorm(c)
    c = c.astype(np.float32)
    c = torch.tensor(c)
    c = c.reshape(1, 1, 512, 128)  # 数量  通道数
    c = c.to('cuda')
    out = net(c)
    out = out.cpu()
    # out = np.asarray(out)
    out = out.detach().numpy()
    aaa = out.reshape(128, 128)
    aaa = aaa.T
    return aaa
c = scipy.io.loadmat(roottest + 'vessel_sensor_data_100_100.mat')['sensor_data']
aaa = testpre(c)
num11 = aaa.flatten()
c = scipy.io.loadmat(roottest + 'vessel_true_100_100.mat')['BV2']
c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
num22 = c.flatten()
p1 = pearsonr(num11, num22)[0]
p1 = round(p1, 2)
pc_iters.append(p1)
p2 = round(psnr(num11, num22), 2)
psnr_iters.append(p2)
np.save('pc_iters.npy', pc_iters)
np.save('psnr_iters.npy', psnr_iters)
# '''