import torch
import torch.nn as nn
import torch.nn.functional as F
from mylib import Bluestack
class soft_mask1(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(soft_mask1,self).__init__()
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = Bluestack(in_ch,in_ch)

        self.down2 = nn.MaxPool2d(2)
        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch, in_ch)
        self.up1 = nn.Upsample(scale_factor=2)

        self.conv4 = Bluestack(in_ch, in_ch)

        self.conv5 = Bluestack(in_ch, in_ch)
        self.up2 = nn.Upsample(scale_factor=2)

        self.conv6 = Bluestack(in_ch,in_ch)

        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        identity = x
        x = self.down1(x)
        x = self.conv1(x)

        x1 = self.up1(self.conv3(self.conv2(self.down2(x))))
        x2 = self.conv4(x)

        x = self.conv5(x1 + x2)
        x = self.up2(x)

        temp = self.conv6(identity) + x
        out = self.relu(self.bn(temp))
        return out
class module1(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(module1,self).__init__()
        self.conv1 = Bluestack(in_ch,in_ch)

        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch,in_ch)

        self.conv4 = soft_mask1(in_ch,in_ch)

        self.conv5 = Bluestack(in_ch, in_ch)
    def forward(self, x):
        x1 = self.conv1(x)

        xup = self.conv3(self.conv2(x1))

        xdown = self.conv4(x1)

        x = xup*xdown
        x = x + xup
        x = self.conv5(x)

        return x
class soft_mask2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(soft_mask2,self).__init__()
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = Bluestack(in_ch,in_ch)

        self.down2 = nn.MaxPool2d(2)
        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch, in_ch)
        self.up1 = nn.Upsample(scale_factor=2)

        self.conv4 = Bluestack(in_ch, in_ch)

        self.conv5 = Bluestack(in_ch, in_ch)
        self.up2 = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)

        x1 = self.up1(self.conv3(self.conv2(self.down2(x))))
        x2 = self.conv4(x)

        x = self.conv5(x1 + x2)
        x = self.up2(x)
        out = self.relu(self.bn(x))
        return out
class module2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(module2,self).__init__()
        self.conv1 = Bluestack(in_ch,in_ch)

        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch,in_ch)

        self.conv4 = soft_mask2(in_ch,in_ch)

        self.conv5 = Bluestack(in_ch, in_ch)
    def forward(self, x):
        x1 = self.conv1(x)

        xup = self.conv3(self.conv2(x1))

        xdown = self.conv4(x1)

        x = xup*xdown
        x = x + xup
        x = self.conv5(x)

        return x
class soft_mask3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(soft_mask3,self).__init__()
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = Bluestack(in_ch,in_ch)

        self.down2 = nn.MaxPool2d(2)
        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch, in_ch)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv5 = Bluestack(in_ch, in_ch)
        self.up2 = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)

        x1 = self.up1(self.conv3(self.conv2(self.down2(x))))

        x = self.conv5(x1)
        x = self.up2(x)
        out = self.relu(self.bn(x))
        return out
class module3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(module3,self).__init__()
        self.conv1 = Bluestack(in_ch,in_ch)

        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = Bluestack(in_ch,in_ch)

        self.conv4 = soft_mask3(in_ch,in_ch)

        self.conv5 = Bluestack(in_ch, in_ch)
    def forward(self, x):
        x1 = self.conv1(x)

        xup = self.conv3(self.conv2(x1))

        xdown = self.conv4(x1)

        x = xup*xdown
        x = x + xup
        x = self.conv5(x)

        return x
class total(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(total,self).__init__()
        self.convbegin = nn.Conv2d(1,in_ch,3,1,1)

        self.conv1 = module1(in_ch,in_ch)
        self.conv2 = Bluestack(in_ch,in_ch)
        self.conv3 = module2(in_ch,in_ch)
        self.conv4 = Bluestack(in_ch,in_ch)
        self.conv5 = module3(in_ch, in_ch)
        self.conv6 = Bluestack(in_ch, in_ch)

        self.convend = nn.Conv2d(in_ch,1,3,1,1)
    def forward(self, x):
        x = self.convbegin(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.convend(x)
        return x
dd = torch.randn(2,1,128,128)
dd = dd.cuda()
djgnet = total(16,3)
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,128,128),1)
