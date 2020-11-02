import torch.nn as nn
import torch.nn.functional as F
import torch
from dense_block2 import _DR_block2
class Bluestack(nn.Module):#must  in_ch = out_ch
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
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out
class BasicConv2d_110(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d_110, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
# '''
class AsymUNet6layers(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32, no_blue = None):
        super(AsymUNet6layers, self).__init__()
        filters = [8,16,32,64,128,256]
        self.conv1_1 = nn.Conv2d(num_input_channels, base_n_features, 3, padding=1)
        self.conv1_2 = _DR_block2(base_n_features, filters[0])
        # self.conv1_3 = Bluestack(base_n_features*2,base_n_features*2)
        self.down1 = nn.MaxPool2d(2)

        self.conv2_1 = _DR_block2(base_n_features*2, filters[1])
        # self.conv2_2 = Bluestack(base_n_features*4, base_n_features*4)
        self.down2 = nn.MaxPool2d(2)

        self.conv3_1 = _DR_block2(base_n_features * 4, filters[2])
        # self.conv3_2 = Bluestack(base_n_features * 8, base_n_features * 8)
        self.down3 = nn.MaxPool2d(2)

        self.conv4_1 = _DR_block2(base_n_features * 8, filters[3])
        # self.conv4_2 = Bluestack(base_n_features * 16, base_n_features * 16)
        self.down4 = nn.MaxPool2d(2)

        self.conv5_1 = _DR_block2(base_n_features * 16, filters[4])
        self.conv5_2 = Bluestack(base_n_features * 32, base_n_features * 32)
        # self.down5 = nn.MaxPool2d(2)

        # self.conv6_1 = _DR_block2(base_n_features * 32, filters[5])

        # self.conv6_5 = Bluestack(base_n_features * 64, base_n_features * 64)
        # self.conv6_5 = BasicConv2d_110(base_n_features * 64, base_n_features * 64)

        # self.up5 = nn.ConvTranspose2d(base_n_features * 64, base_n_features * 32, 4, 2, 1)
        # self.conv5_5 = BasicConv2d_110(base_n_features * 32 + base_n_features * 32,base_n_features * 16)
        # self.conv5_6 = _DR_block2(base_n_features * 16, filters[4])
        # self.conv5_7 = Bluestack(base_n_features * 32, base_n_features * 32)

        self.up4 = nn.ConvTranspose2d(base_n_features * 32, base_n_features * 16, 4, 2, 1)
        self.conv4_5 = BasicConv2d_110(base_n_features * 16 + base_n_features * 16, base_n_features * 8)
        self.conv4_6 = _DR_block2(base_n_features * 8, filters[3])
        # self.conv4_7 = Bluestack(base_n_features * 16, base_n_features * 16)

        self.up3 = nn.ConvTranspose2d(base_n_features * 16, base_n_features * 8, 4, 2, 1)
        self.conv3_5 = BasicConv2d_110(base_n_features * 8 + base_n_features * 8, base_n_features * 4)
        self.conv3_6 = _DR_block2(base_n_features * 4, filters[2])
        # self.conv3_7 = Bluestack(base_n_features * 8, base_n_features * 8)

        self.up2 = nn.ConvTranspose2d(base_n_features * 8, base_n_features * 4, 4, 2, 1)
        self.conv2_5 = BasicConv2d_110(base_n_features * 4 + base_n_features * 4, base_n_features*2)
        self.conv2_6 = _DR_block2(base_n_features*2, filters[1])
        # self.conv2_7 = Bluestack(base_n_features*4, base_n_features*4)

        self.up1 = nn.ConvTranspose2d(base_n_features*4, base_n_features*2, 4, 2, 1)
        self.conv1_5 = BasicConv2d_110(base_n_features*2 + base_n_features*2, base_n_features)
        self.conv1_6 = _DR_block2(base_n_features, filters[0])
        # self.conv1_7 = Bluestack(base_n_features*2, base_n_features*2)

        self.conv1_8 = BasicConv2d_110(base_n_features * 2, base_n_features // 2)

        self.conv1_9 = nn.Conv2d(base_n_features // 2 + 1, 1, 3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.LeakyReLU(inplace=True)

        # Skip connnections:
        self.skip1 = nn.Conv2d(base_n_features * 2, base_n_features * 2, (6, 3), (4, 1), (1, 1))
        self.skip2 = nn.Conv2d(base_n_features * 4, base_n_features * 4, (6, 3), (4, 1), (1, 1))
        self.skip3 = nn.Conv2d(base_n_features * 8, base_n_features * 8, (6, 3), (4, 1), (1, 1))
        self.skip4 = nn.Conv2d(base_n_features * 16, base_n_features * 16, (6, 3), (4, 1), (1, 1))
        self.skip5 = nn.Conv2d(base_n_features * 32, base_n_features * 32, (6, 3), (4, 1), (1, 1))
        # self.skip6 = nn.Conv2d(base_n_features * 64, base_n_features * 64, (6, 3), (4, 1), (1, 1))
        self.skip7 = nn.Conv2d(num_input_channels, 1, (6, 3), (4, 1), (1, 1))
    def forward(self, x):
        s7 = self.skip7(x)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # s1 = x = self.conv1_3(x)
        s1 = self.skip1(x)
        x = self.down1(x)

        x = self.conv2_1(x)
        # s2 = x = self.conv2_2(x)
        s2 = self.skip2(x)
        x = self.down2(x)

        x = self.conv3_1(x)
        # s3 = x = self.conv3_2(x)
        s3 = self.skip3(x)
        x = self.down3(x)

        x = self.conv4_1(x)
        # s4 = x = self.conv4_2(x)
        s4 = self.skip4(x)
        x = self.down4(x)

        x = self.conv5_1(x)
        # s5 = x = self.conv5_2(x)
        s5 = self.skip5(x)
        # x = self.down5(x)

        # s6 = x = self.conv6_1(x)
        # s6 = self.skip6(x)

        x = self.conv5_2(s5)

        # x = self.up5(s6)
        # x = self.conv5_5(torch.cat((x, s5), 1))
        # x = self.conv5_6(x)
        # x = self.conv5_7(x)

        x = self.up4(x)
        x = self.conv4_5(torch.cat((x, s4), 1))
        x = self.conv4_6(x)
        # x = self.conv4_7(x)

        x = self.up3(x)
        x = self.conv3_5(torch.cat((x, s3), 1))
        x = self.conv3_6(x)
        # x = self.conv3_7(x)

        x = self.up2(x)
        x = self.conv2_5(torch.cat((x, s2), 1))
        x = self.conv2_6(x)
        # x = self.conv2_7(x)

        x = self.up1(x)
        x = self.conv1_5(torch.cat((x, s1), 1))
        x = self.conv1_6(x)
        # x = self.conv1_7(x)
        x = self.conv1_8(x)

        x = torch.cat((x, s7), 1)
        x = self.conv1_9(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
# '''
djgnet = AsymUNet6layers(1)
# dd = torch.randn(2,1,512,128)
# dd = dd.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,512,128),1)