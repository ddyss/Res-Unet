import torch.nn as nn
import torch.nn.functional as F
import torch
# '''测试没有top skip 的结果
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
        # self.dropout = nn.Dropout2d(0.5)
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        # out = self.dropout(out)
        return out
class AsymUNet6layers(nn.Module):# from 4 big layers to 6 big layers
    def __init__(self, num_input_channels, base_n_features=32, no_blue = None):  # 16 #24ist auch gut):
        super(AsymUNet6layers, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, base_n_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_n_features)
        if no_blue:
            self.conv2 = nn.Conv2d(base_n_features, base_n_features, 3, padding=1)
        else:
            self.conv2 = Bluestack(base_n_features, base_n_features)
        self.bn2 = nn.BatchNorm2d(base_n_features)
        self.down1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(base_n_features, base_n_features * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_n_features * 2)
        if no_blue:
            self.conv4 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        else:
            self.conv4 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.bn4 = nn.BatchNorm2d(base_n_features * 2)
        self.down2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(base_n_features * 2, base_n_features * 4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_n_features * 4)
        if no_blue:
            self.conv6 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        else:
            self.conv6 = Bluestack(base_n_features * 4, base_n_features * 4)
        self.bn6 = nn.BatchNorm2d(base_n_features * 4)
        self.down3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(base_n_features * 4, base_n_features * 8, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(base_n_features * 8)
        if no_blue:
            self.conv15 = nn.Conv2d(base_n_features * 8, base_n_features * 8, 3, padding=1)
        else:
            self.conv15 = Bluestack(base_n_features * 8, base_n_features * 8)
        self.bn15 = nn.BatchNorm2d(base_n_features * 8)
        self.down4 = nn.MaxPool2d(2)

        self.conv16 = nn.Conv2d(base_n_features * 8, base_n_features * 16, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(base_n_features * 16)
        if no_blue:
            self.conv19 = nn.Conv2d(base_n_features * 16, base_n_features * 16, 3, padding=1)
        else:
            self.conv19 = Bluestack(base_n_features * 16, base_n_features * 16)
        self.bn19 = nn.BatchNorm2d(base_n_features * 16)
        self.down5 = nn.MaxPool2d(2)

        self.conv20 = nn.Conv2d(base_n_features * 16, base_n_features * 32, 3, padding=1)
        self.bn20 = nn.BatchNorm2d(base_n_features * 32)
        if no_blue:
            self.conv21 = nn.Conv2d(base_n_features * 32, base_n_features * 32, 3, padding=1)
        else:
            self.conv21 = Bluestack(base_n_features * 32, base_n_features * 32)
        self.bn21 = nn.BatchNorm2d(base_n_features * 32)

        self.up5 = nn.Upsample(scale_factor=2)
        # self.up5 = nn.ConvTranspose2d(base_n_features * 32 , base_n_features * 32,4,2,1)
        self.conv22 = nn.Conv2d(base_n_features * 32 + base_n_features * 16, base_n_features * 16, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(base_n_features * 16)
        if no_blue:
            self.conv17 = nn.Conv2d(base_n_features * 16, base_n_features * 16, 3, padding=1)
        else:
            self.conv17 = Bluestack(base_n_features * 16, base_n_features * 16)
        self.bn17 = nn.BatchNorm2d(base_n_features * 16)

        self.up4 = nn.Upsample(scale_factor=2)
        # self.up4 = nn.ConvTranspose2d(base_n_features * 16, base_n_features * 16, 4, 2, 1)
        self.conv18 = nn.Conv2d(base_n_features * 16 + base_n_features * 8, base_n_features * 8, 3, padding=1)
        self.bn18 = nn.BatchNorm2d(base_n_features * 8)
        if no_blue:
            self.conv8 = nn.Conv2d(base_n_features * 8, base_n_features * 8, 3, padding=1)
        else:
            self.conv8 = Bluestack(base_n_features * 8, base_n_features * 8)
        self.bn8 = nn.BatchNorm2d(base_n_features * 8)

        self.up1 = nn.Upsample(scale_factor=2)
        # self.up1 = nn.ConvTranspose2d(base_n_features * 8, base_n_features * 8, 4, 2, 1)
        self.conv9 = nn.Conv2d(base_n_features * 8 + base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(base_n_features * 4)
        if no_blue:
            self.conv10 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        else:
            self.conv10 = Bluestack(base_n_features * 4, base_n_features * 4)
        self.bn10 = nn.BatchNorm2d(base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        # self.up2 = nn.ConvTranspose2d(base_n_features * 4, base_n_features * 4, 4, 2, 1)
        self.conv11 = nn.Conv2d(base_n_features * 4 + base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(base_n_features * 2)
        if no_blue:
            self.conv12 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        else:
            self.conv12 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.bn12 = nn.BatchNorm2d(base_n_features * 2)

        self.up3 = nn.Upsample(scale_factor=2)
        # self.up3 = nn.ConvTranspose2d(base_n_features * 2, base_n_features * 2, 4, 2, 1)
        self.conv13 = nn.Conv2d(base_n_features * 2 + base_n_features, base_n_features, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(base_n_features)
        self.conv14 = nn.Conv2d(base_n_features * 1, 1, 3, padding=1)

        # self.conv23 = nn.Conv2d(base_n_features//2 + 1, 1, 3, padding=1)

        # Skip connnections:
        self.skip1 = nn.Conv2d(base_n_features, base_n_features, (6, 3), (4, 1), (1, 1))
        self.skip2 = nn.Conv2d(base_n_features * 2, base_n_features * 2, (6, 3), (4, 1), (1, 1))
        self.skip3 = nn.Conv2d(base_n_features * 4, base_n_features * 4, (6, 3), (4, 1), (1, 1))
        self.skip4 = nn.Conv2d(base_n_features * 8, base_n_features * 8, (6, 3), (4, 1), (1, 1))
        self.skip5 = nn.Conv2d(base_n_features * 16, base_n_features * 16, (6, 3), (4, 1), (1, 1))
        self.skip6 = nn.Conv2d(base_n_features * 32, base_n_features * 32, (6, 3), (4, 1), (1, 1))
        # self.skip7 = nn.Conv2d(num_input_channels, 1, (6, 3), (4, 1), (1, 1))
    def forward(self, x):
        # s7 = self.skip7(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        s1 = x = F.leaky_relu(self.bn2(self.conv2(x)))
        s1 = self.skip1(s1)
        x = self.down1(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        s2 = x = F.leaky_relu(self.bn4(self.conv4(x)))
        s2 = self.skip2(s2)
        x = self.down2(x)

        x = F.leaky_relu(self.bn5(self.conv5(x)))
        s3 = x = F.leaky_relu(self.bn6(self.conv6(x)))
        s3 = self.skip3(s3)
        x = self.down3(x)

        x = F.leaky_relu(self.bn7(self.conv7(x)))
        s4 = x = F.leaky_relu(self.bn15(self.conv15(x)))
        s4 = self.skip4(s4)
        x = self.down4(x)

        x = F.leaky_relu(self.bn16(self.conv16(x)))
        s5 = x = F.leaky_relu(self.bn19(self.conv19(x)))
        s5 = self.skip5(s5)
        x = self.down5(x)

        s6 = x = F.leaky_relu(self.bn20(self.conv20(x)))
        s6 = self.skip6(s6)
        x = F.leaky_relu(self.bn21(self.conv21(s6)))

        x = self.up5(x)
        x = F.leaky_relu(self.bn22(self.conv22(torch.cat((x, s5), 1))))
        x = F.leaky_relu(self.bn17(self.conv17(x)))

        x = self.up4(x)
        x = F.leaky_relu(self.bn18(self.conv18(torch.cat((x, s4), 1))))
        x = F.leaky_relu(self.bn8(self.conv8(x)))

        x = self.up1(x)
        x = F.leaky_relu(self.bn9(self.conv9(torch.cat((x, s3), 1))))
        x = F.leaky_relu(self.bn10(self.conv10(x)))

        x = self.up2(x)
        x = F.leaky_relu(self.bn11(self.conv11(torch.cat((x, s2), 1))))
        x = F.leaky_relu(self.bn12(self.conv12(x)))

        x = self.up3(x)
        x = F.leaky_relu(self.bn13(self.conv13(torch.cat((x, s1), 1))))
        x = F.leaky_relu(self.conv14(x))
        # x = torch.cat((x, s7), 1)
        # x = self.conv23(x)
        return x
# '''
djgnet = AsymUNet6layers(1,no_blue = False)
# djgnet = djgnet.to('cuda')
# dd = torch.randn(1,1,512,128)
# dd = dd.cuda()
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,512,128),1)