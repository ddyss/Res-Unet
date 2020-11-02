# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
# 已经用仿真数据测试过，进过这组ChannelNet后拼接结果还是类似sensor_data的形状
#两种：一种2048直接卷积到128；另一种2048卷积到32再反卷到128
class CNN_1(nn.Module): #batch 1 2048-----batch 32 128
    def __init__(self):
        super(CNN_1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv1d(1, 8, 3, 1, 1)#
        self.conv1_2 = nn.Conv1d(1, 8, 5, 1, 2)
        self.conv1_3 = nn.Conv1d(1, 8, 7, 1, 3)
        self.conv1_4 = nn.Conv1d(1, 8, 9, 1, 4)

        self.norm1 = nn.BatchNorm1d(32) # batch 32 1024
        self.pool1 = nn.MaxPool1d(2, return_indices=True) # batch 32 512

        self.conv2_1 = nn.Conv1d(32, 16, 3,1,1) #
        self.conv2_2 = nn.Conv1d(32, 16, 5,1,2) # batch 32 512

        self.norm2 = nn.BatchNorm1d(32)   #batch 32 512
        self.pool2 = nn.MaxPool1d(2, return_indices=True) # batch 32 256

    def forward(self, x):   # batch x 1 x raw_feature
        encoded = torch.cat([self.conv1_1(x),self.conv1_2(x),self.conv1_3(x),self.conv1_4(x)],1)
        encoded = self.relu(self.norm1(encoded))
        encoded, indices1 = self.pool1(encoded)

        encoded = torch.cat([self.conv2_1(encoded),self.conv2_2(encoded)],1)
        encoded = self.relu(self.norm2(encoded))
        encoded, indices2 = self.pool2(encoded)
        return encoded

class AE_1(nn.Module):
    def __init__(self, hidden_size):
        super(AE_1, self).__init__()
        self.hidden_size = hidden_size
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(2, stride=2, return_indices=True)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=2)
        self.pool2 = nn.MaxPool1d(2, stride=1, return_indices=True)
        self.fc1 = nn.Linear(8 * 256, self.hidden_size)

        self.fc2 = nn.Linear(self.hidden_size, 8 * 256)
        self.unpool2 = nn.MaxUnpool1d(2, stride=1)
        self.deconv2 = nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=2)
        self.unpool1 = nn.MaxUnpool1d(2, stride=2)
        self.deconv1 = nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=2)

    def forward(self, x):        # shape: batch x 1 x raw_featuture
        encoded = self.relu(self.conv1(x))
        encoded_size1 = encoded.size()
        encoded, indices1 = self.pool1(encoded)
        encoded = self.relu(self.conv2(encoded))
        encoded_size2 = encoded.size()
        encoded, indices2 = self.pool2(encoded)
        encoded_size3 = encoded.size()

        encoded = self.relu(self.fc1(encoded.view(encoded.size(0), -1)))  # batch x hidden_size

        decoded = self.relu(self.fc2(encoded))
        decoded = decoded.view(encoded_size3)

        decoded = self.unpool2(decoded, indices2)  # new added due to some reasons
        # #        decoded = self.unpool2(decoded, indices2, output_size=encoded_size2)
        # decoded = self.relu(self.deconv2(decoded))
        # decoded = self.unpool1(decoded, indices1)  # new added due to some reasons
        # #        decoded = self.unpool1(decoded, indices1, output_size=encoded_size1)
        # decoded = self.sigmoid(self.deconv1(decoded))
        # decoded = decoded.view(decoded.size(0), -1)  # new added due to some reasons

        return decoded

class ChannelNet(nn.Module):
    def __init__(self, hidden_size):
        super(ChannelNet, self).__init__()
        self.hidden_size = hidden_size
        self.relu = nn.ReLU(inplace=True)
        self.cnn1 = CNN_1()

    def forward(self, x):  # shape: batch x channel x 512 x 128
        encoded = torch.zeros(x.size(0), 32, self.hidden_size, x.size(3))  # batch x channel x hidden_size
        for i in range(x.size(3)):
            out1 = x[:,:,:,i].contiguous()
            out1 = out1.view(out1.size(0), -1, out1.size(2))  # batch*1*raw_feature
            encoded[:,:,:,i] = self.cnn1(out1)
        return encoded
# dd=torch.randn(4,1,512,128)
# dd=dd.cuda()
# djgnet=ChannelNet(128)
# djgnet=djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)

# from torchsummary import summary
# summary(djgnet,(1,2048,128),1)   # 16217 MB