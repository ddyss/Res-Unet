import torch
import torch.nn as nn
import torch.nn.functional as F
class Singleconv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Singleconv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
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
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out
class encoded_skip_abd4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encoded_skip_abd4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 6,4, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class encoded_skip_abd8(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encoded_skip_abd8, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 10,8, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class decoded_skip_abd4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoded_skip_abd4, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 6,4, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class decoded_skip_abd8(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoded_skip_abd8, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 10,8, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class skip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(skip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (6, 3), (4, 1), 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class Net(nn.Module):
    def __init__(self,base_channel=16):
        super(Net, self).__init__()
        self.conv1 = Singleconv2d(1,base_channel)
        self.conv1_blue = Bluestack(base_channel,base_channel)
        self.down1 = nn.MaxPool2d(2)
        self.skip13 = encoded_skip_abd4(base_channel,base_channel)
        self.skip14 = encoded_skip_abd8(base_channel, base_channel)
        self.skip18 = skip(base_channel, base_channel)

        self.conv2 = Singleconv2d(base_channel, base_channel*2)
        self.conv2_blue = Bluestack(base_channel*2, base_channel*2)
        self.down2 = nn.MaxPool2d(2)
        self.skip24 = encoded_skip_abd4(base_channel*2, base_channel * 2)
        self.skip27 = skip(base_channel * 2, base_channel * 2)

        self.conv3 = Singleconv2d(base_channel*2 + base_channel, base_channel * 4)
        self.conv3_blue = Bluestack(base_channel*4, base_channel*4)
        self.down3 = nn.MaxPool2d(2)
        self.skip36 = skip(base_channel * 4, base_channel * 4)

        self.conv4 = Singleconv2d(base_channel * 4 + base_channel * 2 + base_channel, base_channel * 8)
        self.skip45 = skip(base_channel * 8, base_channel * 8)
        #decoded
        self.conv5_blue = Bluestack(base_channel * 8, base_channel * 8)
        self.up3 = nn.Upsample(scale_factor=2)
        # self.skip57 = decoded_skip_abd4(base_channel*16, base_channel * 8)
        # self.skip58 = decoded_skip_abd8(base_channel*16, base_channel * 8)

        self.conv6 = Singleconv2d(base_channel * 8 + base_channel * 4, base_channel * 4)
        self.conv6_blue = Bluestack(base_channel * 4, base_channel * 4)
        self.up2 = nn.Upsample(scale_factor=2)
        # self.skip68 = decoded_skip_abd4(base_channel * 8, base_channel * 4)

        self.conv7 = Singleconv2d(base_channel * 4 + base_channel * 2, base_channel * 2)
        self.conv7_blue = Bluestack(base_channel * 2, base_channel * 2)
        self.up1 = nn.Upsample(scale_factor=2)

        self.conv8 = Singleconv2d(base_channel * 2 + base_channel, base_channel)
        # self.conv8_blue = Bluestack(base_channel * 8, base_channel * 8)

        self.convfinal = Singleconv2d(base_channel, 1)
    def forward(self, x):
        identity = x
        x_1 = self.conv1_blue(self.conv1(x))
        x_d1 = self.down1(x_1)
        x_s13 = self.skip13(x_1)
        x_s14 = self.skip14(x_1)
        x_s18 = self.skip18(x_1)

        x_2 = self.conv2_blue(self.conv2(x_d1))
        x_d2 = self.down2(x_2)
        x_s24 = self.skip24(x_2)
        x_s27 = self.skip27(x_2)

        x_3 = self.conv3_blue(self.conv3(torch.cat([x_d2,x_s13],1)))
        x_d3 = self.down3(x_3)
        x_s36 = self.skip36(x_3)

        x_4 = self.conv4(torch.cat([x_d3, x_s14,x_s24], 1))
        x_s45 = self.skip45(x_4)

        x_5 = self.conv5_blue(x_s45)
        x_up3 = self.up3(x_5)
        # x_s57 = self.skip57(x_5)
        # x_s58 = self.skip58(x_5)

        x_6 = self.conv6_blue(self.conv6(torch.cat([x_up3, x_s36], 1)))
        x_up2 = self.up2(x_6)
        # x_s68 = self.skip68(x_6)

        x_7 = self.conv7_blue(self.conv7(torch.cat([x_up2, x_s27], 1)))
        x_up1 = self.up1(x_7)

        x_8 = self.conv8(torch.cat([x_up1, x_s18], 1))

        out = self.convfinal(x_8)
        return out

dd = torch.randn(2,1,512,128)
dd = dd.cuda()
djgnet = Net()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,512,128),1)
