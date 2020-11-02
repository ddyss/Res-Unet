# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io




# djgnet=ChannelNet(128)
# djgnet=djgnet.to('cuda')
# dd=torch.randn(4,1,2048,128)
# dd=dd.cuda()
# yy = djgnet(dd)
# print(yy.shape)

# from torchsummary import summary
# summary(djgnet,(1,2030,128),2)