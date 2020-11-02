from dataloader2 import train_loader#,valid_loader
import torch.nn as nn
from torch import optim
import torch
from timeit import default_timer as timer
import numpy as np
import argparse
from model2 import djgnet

net=nn.DataParallel(djgnet).to('cuda')
model_name='unet_paper.pkl'
loss_func=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=0.0001,weight_decay=0.00001)
trainloss_total=[]
# validloss_total=[]
bestloss = np.Inf
for epoch in range(50):
    start1=timer()
    trainloss=0
    # validloss=0
    net.train()
    for i,(data,qqtarget,target)in enumerate(train_loader):
        data=data.cuda()
        qqtarget=qqtarget.cuda()
        target=target.cuda()
        out=net(data)
        loss1 = loss_func(out[0],qqtarget)
        loss2 = loss_func(out[1], target)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        progress=round(100*(i+1)/len(train_loader),2)
        time=round(timer()-start1,2)
        # print(epoch, ' ', i, ':', loss.item(), end='\n')
        print(epoch,i, ':', loss.item(),' ',progress,'% complete ',time,'second passed',end='\r')
    # with torch.no_grad():
    #     net.eval()
    #     start2=timer()
    #     for i,(data,target) in enumerate(valid_loader):
    #         data=data.cuda()
    #         target=target.cuda()
    #         out=net(data)
    #         loss=loss_func(out,target)
    #         validloss+=loss.item()
    #         progress = round(100 * (i + 1) / len(valid_loader), 2)
    #         time = round(timer() - start2, 2) #对浮点数进行4舍5入取值，但不总是，保留几位小数,不写的话默认保留到整数。
    #         # print(epoch, ' ', i, ':', loss.item(), end='\n')
    #         print(epoch,i, ':', loss.item(),' ',progress, '% complete ', time, 'second passed', end='\r')
    trainloss=trainloss/len(train_loader.dataset)
    # validloss=validloss/len(valid_loader.dataset)

    trainloss_total.append(trainloss)
    np.save('trainloss.npy', trainloss_total)
    torch.save(net, model_name)
    torch.save(net.state_dict(), 'unet_paper_params.pkl')

    # validloss_total.append(validloss)
    # np.save('validloss.npy', validloss_total)

    # if validloss<bestloss:
    #     torch.save(net,model_name)
    #     bestloss=validloss
    #     bestepoch=epoch
    #
    # trainloss = round(trainloss, 6)
    # validloss = round(validloss, 6)
    # print('epoch:',epoch,'trainloss:',trainloss,'validloss:',validloss)

# print('best epoch:',bestepoch,'  ','bestloss:',bestloss)

