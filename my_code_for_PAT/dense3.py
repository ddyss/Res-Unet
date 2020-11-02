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
class encoded_skip_abd16(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encoded_skip_abd16, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 18,16, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class encoded_skip_abd32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encoded_skip_abd32, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 34,32, 1),
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
class decoded_skip_abd16(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoded_skip_abd16, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 18,16, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class decoded_skip_abd32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoded_skip_abd32, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 34,32, 1),
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
    def __init__(self,base_channel=8):
        super(Net, self).__init__()
        self.conv1_1 = Singleconv2d(1,base_channel)
        self.conv1_2 = Bluestack(base_channel, base_channel)
        self.down1 = nn.MaxPool2d(2)
        self.skip13 = encoded_skip_abd4(base_channel,base_channel)
        self.skip14 = encoded_skip_abd8(base_channel, base_channel)
        self.skip19 = encoded_skip_abd16(base_channel, base_channel)
        self.skip1_10 = encoded_skip_abd32(base_channel, base_channel)
        self.skip18 = skip(base_channel, base_channel)

        self.conv2_1 = Singleconv2d(base_channel, base_channel*2)
        self.conv2_2 = Bluestack(base_channel*2, base_channel * 2)
        self.down2 = nn.MaxPool2d(2)
        self.skip24 = encoded_skip_abd4(base_channel*2, base_channel * 2)
        self.skip29 = encoded_skip_abd8(base_channel*2, base_channel*2)
        self.skip2_10 = encoded_skip_abd16(base_channel*2, base_channel*2)
        self.skip27 = skip(base_channel * 2, base_channel * 2)

        self.conv3_1 = Singleconv2d(base_channel*2 + base_channel, base_channel * 4)
        self.conv3_2 = Bluestack(base_channel * 4, base_channel * 4)
        self.down3 = nn.MaxPool2d(2)
        self.skip39 = encoded_skip_abd4(base_channel*4, base_channel*4)
        self.skip3_10 = encoded_skip_abd8(base_channel*4, base_channel*4)
        self.skip36 = skip(base_channel * 4, base_channel * 4)

        self.conv4_1 = Singleconv2d(base_channel * 4 + base_channel * 2 + base_channel, base_channel * 8)
        self.conv4_2 = Bluestack(base_channel * 8, base_channel * 8)
        self.down4 = nn.MaxPool2d(2)
        self.skip4_10 = encoded_skip_abd4(base_channel * 8, base_channel * 8)
        self.skip45 = skip(base_channel * 8, base_channel * 8)

        self.conv9_1 = Singleconv2d(base_channel * 8 + base_channel * 4 + base_channel * 2 + base_channel, base_channel * 16)
        self.conv9_2 = Bluestack(base_channel * 16, base_channel * 16)
        self.down9 = nn.MaxPool2d(2)
        self.skip9_12 = skip(base_channel * 16, base_channel * 16)

        self.conv10_1 = Singleconv2d(base_channel * 16 + base_channel * 8 + base_channel * 4 + base_channel * 2 + base_channel, base_channel * 32)
        # self.conv10_2 = Bluestack(base_channel * 32, base_channel * 32)
        self.skip10_11 = skip(base_channel * 32, base_channel * 32)
        #decoded
        self.conv11_1 = Singleconv2d(base_channel * 32, base_channel * 32)
        # self.conv11_2 = Bluestack(base_channel * 32, base_channel * 32)
        self.up5 = nn.Upsample(scale_factor=2)
        self.skip11_5 = decoded_skip_abd4(base_channel * 32, base_channel * 32)
        self.skip11_6 = decoded_skip_abd8(base_channel * 32, base_channel * 32)
        self.skip11_7 = decoded_skip_abd16(base_channel * 32, base_channel * 32)
        self.skip11_8 = decoded_skip_abd32(base_channel * 32, base_channel * 32)

        self.conv12_1 = Singleconv2d(base_channel * 32 + base_channel * 16, base_channel * 16)
        self.conv12_2 = Bluestack(base_channel * 16, base_channel * 16)
        self.up4 = nn.Upsample(scale_factor=2)
        self.skip12_6 = decoded_skip_abd4(base_channel * 16, base_channel * 16)
        self.skip12_7 = decoded_skip_abd8(base_channel * 16, base_channel * 16)
        self.skip12_8 = decoded_skip_abd16(base_channel * 16, base_channel * 16)

        self.conv5_1 = Singleconv2d(base_channel * 32 + base_channel * 16 + base_channel * 8, base_channel * 8)
        self.conv5_2 = Bluestack(base_channel * 8, base_channel * 8)
        self.up3 = nn.Upsample(scale_factor=2)
        self.skip57 = decoded_skip_abd4(base_channel*8, base_channel * 8)
        self.skip58 = decoded_skip_abd8(base_channel*8, base_channel * 8)

        self.conv6_1 = Singleconv2d(base_channel * 32 + base_channel * 16 + base_channel * 8 + base_channel * 4, base_channel * 4)
        self.conv6_2 = Bluestack(base_channel * 4, base_channel * 4)
        self.up2 = nn.Upsample(scale_factor=2)
        self.skip68 = decoded_skip_abd4(base_channel * 4, base_channel * 4)

        self.conv7_1 = Singleconv2d(base_channel * 32 + base_channel * 16 + base_channel * 8 + base_channel * 4 + base_channel * 2, base_channel * 2)
        self.conv7_2 = Bluestack(base_channel * 2, base_channel * 2)
        self.up1 = nn.Upsample(scale_factor=2)

        self.conv8_1 = Singleconv2d(base_channel * 32 + base_channel * 16 + base_channel * 8 + base_channel * 4 + base_channel * 2 + base_channel, base_channel)
        self.conv8_2 = Bluestack(base_channel, base_channel)

        self.convfinal = Singleconv2d(base_channel, 1)
    def forward(self, x):
        identity = x

        x_1 = self.conv1_2(self.conv1_1(x))
        x_d1 = self.down1(x_1)
        x_s13 = self.skip13(x_1)
        x_s14 = self.skip14(x_1)
        x_s19 = self.skip19(x_1)
        x_s1_10 = self.skip1_10(x_1)
        x_s18 = self.skip18(x_1)

        x_2 = self.conv2_2(self.conv2_1(x_d1))
        x_d2 = self.down2(x_2)
        x_s24 = self.skip24(x_2)
        x_s29 = self.skip29(x_2)
        x_s2_10 = self.skip2_10(x_2)
        x_s27 = self.skip27(x_2)

        x_3 = self.conv3_2(self.conv3_1(torch.cat([x_d2,x_s13],1)))
        x_d3 = self.down3(x_3)
        x_s39 = self.skip39(x_3)
        x_s3_10 = self.skip3_10(x_3)
        x_s36 = self.skip36(x_3)

        x_4 = self.conv4_2(self.conv4_1(torch.cat([x_d3, x_s14, x_s24], 1)))
        x_d4 = self.down4(x_4)
        x_s4_10 = self.skip4_10(x_4)
        x_s45 = self.skip45(x_4)

        x_9 = self.conv9_2(self.conv9_1(torch.cat([x_d4, x_s19, x_s29, x_s39], 1)))
        x_d5 = self.down4(x_9)
        x_s9_12 = self.skip9_12(x_9)

        x_10 = self.conv10_1(torch.cat([x_d5, x_s1_10, x_s2_10, x_s3_10, x_s4_10], 1))
        x_s10_11 = self.skip10_11(x_10)
        #decoded
        x_11 = self.conv11_1(x_s10_11)
        x_up5 = self.up5(x_11)
        x_s11_5 = self.skip11_5(x_11)
        x_s11_6 = self.skip11_6(x_11)
        x_s11_7 = self.skip11_7(x_11)
        x_s11_8 = self.skip11_8(x_11)

        x_12 = self.conv12_2(self.conv12_1(torch.cat([x_up5, x_s9_12], 1)))
        x_up4 = self.up5(x_12)
        x_s12_6 = self.skip12_6(x_12)
        x_s12_7 = self.skip12_7(x_12)
        x_s12_8 = self.skip12_8(x_12)

        x_5 = self.conv5_2(self.conv5_1(torch.cat([x_up4, x_s11_5, x_s45], 1)))
        x_up3 = self.up3(x_5)
        x_s57 = self.skip57(x_5)
        x_s58 = self.skip58(x_5)

        x_6 = self.conv6_2(self.conv6_1(torch.cat([x_up3, x_s12_6, x_s11_6, x_s36], 1)))
        x_up2 = self.up2(x_6)
        x_s68 = self.skip68(x_6)

        x_7 = self.conv7_2(self.conv7_1(torch.cat([x_up2, x_s57, x_s12_7, x_s11_7, x_s27], 1)))
        x_up1 = self.up1(x_7)

        x_8 = self.conv8_2(self.conv8_1(torch.cat([x_up1, x_s68, x_s58, x_s12_8, x_s11_8, x_s18], 1)))

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
