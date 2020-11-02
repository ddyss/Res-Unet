import torch.nn as nn
import torch

class Bluestack(nn.Module):#resblock_add, must  in_ch = out_ch
    def __init__(self, in_ch, out_ch):
        super(Bluestack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # self.dropout = nn.Dropout2d(0.5)
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        # out = self.dropout(out)
        return out
class Ad2(nn.Module):#输出channel=2*out_ch
    def __init__(self, in_ch, out_ch):
        super(Ad2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (4, 3), (2, 1), 1),
            nn.BatchNorm2d(out_ch), #
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        # self.bn = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, (4, 3), (2, 1), 1)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = torch.cat([self.conv(x),self.skip(x)],1)
        return out
class TotalNet(nn.Module):
    def __init__(self,in_channels,base_channels=32):
        super(TotalNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,base_channels,(3,1),1,(1,0))
        self.conv2_1 = Bluestack(base_channels,base_channels)
        self.conv2_2 = Bluestack(base_channels,base_channels)
        self.conv2_3 = Bluestack(base_channels,base_channels)
        self.down1 = nn.MaxPool2d((2,1))
        self.conv3 = nn.Conv2d(base_channels, base_channels*2, (3,1),1,(1,0))
        self.conv4_1 = Bluestack(base_channels*2,base_channels*2)
        self.conv4_2 = Bluestack(base_channels*2,base_channels*2)
        self.conv4_3 = Bluestack(base_channels*2,base_channels*2)
        self.down2 = nn.MaxPool2d((2, 1))
        self.conv10 = nn.Conv2d(base_channels*2, base_channels*4, (3,1),1,(1,0))
        self.conv5_1 = Bluestack(base_channels*4, base_channels*4)
        self.conv5_2 = Bluestack(base_channels*4, base_channels*4)
        self.conv5_3 = Bluestack(base_channels*4, base_channels*4)
        self.conv6 = nn.Conv2d(base_channels*4, 1, 1,1,0)
        self.fc1 = nn.Linear(16384,16384)
    def forward(self, x):
        batch = x.size(0)
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.down1(x)
        x = self.conv3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.down2(x)
        x = self.conv10(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv6(x)
        x = x.view(batch,-1)
        x = self.fc1(x)
        x = x.view(batch,1,128,128)

        return x
djgnet = TotalNet(1)
# djgnet = djgnet.cuda()
# dd = torch.randn(2,1,512,128)
# dd = dd.cuda()
# yy = djgnet(dd)
# print(yy.shape)
