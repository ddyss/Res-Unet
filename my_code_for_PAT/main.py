from dataloader_final import train_loader,valid_loader
import torch.nn as nn
from torch import optim
import torch
from timeit import default_timer as timer
import numpy as np
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #"0,1"
#当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能，但会导致不能复现，
torch.backends.cudnn.benchmark = True
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
# from apex import amp
from res_unet_6layers_notop import djgnet

net = nn.DataParallel(djgnet).to('cuda')
# net = djgnet.to('cuda')
model_name='unet_paper.pkl'
loss_func=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=0.0001,weight_decay=0.00001)
# net,optimizer=amp.initialize(net,optimizer,opt_level='O1',verbosity=0)
trainloss_total=[]
# trainloss=[]
validloss_total=[]
# validloss=[]
bestloss = np.Inf
for epoch in range(60):
    start1=timer()
    trainloss=0
    validloss=0
    net.train()
    for i,(data,target)in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        out=net(data)
        loss=loss_func(out,target)
        optimizer.zero_grad()
        # with amp.scale_loss(loss,optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        progress=round(100*(i+1)/len(train_loader),2)
        time=round(timer()-start1,2)
        show = round(loss.item(),6)
        # print(epoch, ' ', i, ':', loss.item(), end='\n')
        print(epoch,i, ':', show,' ',progress,'% complete','  ',time,'second passed',end='\r')
    with torch.no_grad():
        net.eval()
        start2=timer()
        for i,(data,target) in enumerate(valid_loader):
            data=data.cuda()
            # data_f = data_f.cuda()
            target=target.cuda()
            out=net(data)
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

    if validloss<bestloss:
        torch.save(net,model_name)
        # torch.save(net.state_dict(), 'unet_paper_params.pkl')#只取决于net.state_dict()，和pkl.pth后缀无关，而且大小一样
        bestloss=validloss
        bestepoch=epoch

    trainloss = round(trainloss, 6)
    validloss = round(validloss, 6)
    print('epoch:',epoch,'trainloss:',trainloss,'validloss:',validloss)

bestloss = round(bestloss,6)
print('best epoch:',bestepoch,'  ','bestloss:',bestloss)
