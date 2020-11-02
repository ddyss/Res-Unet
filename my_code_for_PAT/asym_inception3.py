import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, base_branch_channels):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, base_branch_channels, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, base_branch_channels, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, base_branch_channels, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, base_branch_channels, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class BasicConv2d_110(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d_110, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class BasicConv2d_def(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d_def, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class BasicConv2d_311(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d_311, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class Singleconv2d_Ad4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Singleconv2d_Ad4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (6,3),(4,1),1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class Singleconv2d_Ad2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Singleconv2d_Ad2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (4,3),(2,1),1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class inception_simple(nn.Module):
    def __init__(self, in_channels, branch_channels,base_channels=16):
        super(inception_simple, self).__init__()
        self.branch1x1 = BasicConv2d_110(in_channels, branch_channels)

        self.branch3x3_1 = BasicConv2d_110(in_channels, base_channels)
        self.branch3x3_2 = BasicConv2d_311(base_channels, branch_channels)
        
        self.branch5x5_1 = BasicConv2d_110(in_channels, base_channels)
        self.branch5x5_2 = BasicConv2d_311(base_channels, base_channels)
        self.branch5x5_3 = BasicConv2d_311(base_channels, branch_channels)
        
        self.branch7x7_1 = BasicConv2d_110(in_channels, base_channels)
        self.branch7x7_2 = BasicConv2d_311(base_channels, base_channels)
        self.branch7x7_3 = BasicConv2d_311(base_channels, base_channels)
        self.branch7x7_4 = BasicConv2d_311(base_channels, branch_channels)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch5x5 = self.branch5x5_3(self.branch5x5_2(self.branch5x5_1(x)))
        branch7x7 = self.branch7x7_4(self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x))))

        outputs = [branch1x1, branch3x3, branch5x5, branch7x7]
        return torch.cat(outputs, 1)
class Bluestack(nn.Module):
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
class AsymUNet6layers(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32, incep=None, no_blue=None):  # 16 #24ist auch gut):
        super(AsymUNet6layers, self).__init__()
        if incep:
            # self.conv1 = inception_simple(num_input_channels, base_n_features // 4)
            self.conv1 = BasicConv2d_311(num_input_channels, base_n_features)
        else:
            self.conv1 = BasicConv2d_311(num_input_channels, base_n_features)
        if no_blue:
            self.conv2 = BasicConv2d_311(base_n_features, base_n_features)
        else:
            self.conv2 = Bluestack(base_n_features, base_n_features)
        self.down1 = nn.MaxPool2d(2)

        if incep:
            # self.conv3 = inception_simple(base_n_features, base_n_features * 2 // 4)
            self.conv3 = BasicConv2d_311(base_n_features, base_n_features * 2)
        else:
            self.conv3 = BasicConv2d_311(base_n_features, base_n_features * 2 )
        if no_blue:
            self.conv4 = BasicConv2d_311(base_n_features * 2, base_n_features * 2 )
        else:
            self.conv4 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.down2 = nn.MaxPool2d(2)

        if incep:
            # self.conv5 = inception_simple(base_n_features * 2, base_n_features * 4 // 4)
            self.conv5 = BasicConv2d_311(base_n_features * 2, base_n_features * 4)
        else:
            self.conv5 = BasicConv2d_311(base_n_features * 2, base_n_features * 4 )
        if no_blue:
            self.conv6 = BasicConv2d_311(base_n_features * 4, base_n_features * 4 )
        else:
            self.conv6 = Bluestack(base_n_features * 4, base_n_features * 4)
        self.down3 = nn.MaxPool2d(2)

        if incep:
            # self.conv7 = inception_simple(base_n_features * 4, base_n_features * 8 // 4)
            self.conv7 = InceptionA(base_n_features * 4,32)
        else:
            self.conv7 = BasicConv2d_311(base_n_features * 4, base_n_features * 8 )
        if no_blue:
            self.conv15 = BasicConv2d_311(base_n_features * 8, base_n_features * 8)
        else:
            self.conv15 = Bluestack(base_n_features * 8, base_n_features * 8)
        self.down4 = nn.MaxPool2d(2)

        if incep:
            # self.conv16 = inception_simple(base_n_features * 8, base_n_features * 16 // 4)
            self.conv16 = InceptionC(base_n_features * 8,128,128)
        else:
            self.conv16 = BasicConv2d_311(base_n_features * 8, base_n_features * 16 )
        if no_blue:
            self.conv19 = BasicConv2d_311(base_n_features * 16, base_n_features * 16 )
        else:
            # self.conv19 = Bluestack(base_n_features * 16, base_n_features * 16)
            self.conv19 = Bluestack(128*4,128*4)
        self.down5 = nn.MaxPool2d(2)

        if incep:
            # self.conv20 = inception_simple(base_n_features * 16, base_n_features * 32//4)
            self.conv20 = InceptionC(128*4,128,256)
        else:
            self.conv20 = BasicConv2d_311(base_n_features * 16, base_n_features * 32 )
        if no_blue:
            self.conv21 = BasicConv2d_311(base_n_features * 32, base_n_features * 32 )
        else:
            # self.conv21 = Bluestack(base_n_features * 32, base_n_features * 32)
            self.conv21 = Bluestack(256 * 4, 256 * 4)

        self.up5 = nn.Upsample(scale_factor=2)
        self.conv22 = BasicConv2d_311(1024 + 512, base_n_features * 16 )
        if no_blue:
            self.conv17 = BasicConv2d_311(base_n_features * 16, base_n_features * 16 )
        else:
            self.conv17 = Bluestack(base_n_features * 16, base_n_features * 16)

        self.up4 = nn.Upsample(scale_factor=2)
        self.conv18 = BasicConv2d_311(512 + base_n_features * 8, base_n_features * 8 )
        if no_blue:
            self.conv8 = BasicConv2d_311(base_n_features * 8, base_n_features * 8 )
        else:
            self.conv8 = Bluestack(base_n_features * 8, base_n_features * 8)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv9 = BasicConv2d_311(base_n_features * 8 + base_n_features * 4, base_n_features * 4 )
        if no_blue:
            self.conv10 = BasicConv2d_311(base_n_features * 4, base_n_features * 4 )
        else:
            self.conv10 = Bluestack(base_n_features * 4, base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv11 = BasicConv2d_311(base_n_features * 4 + base_n_features * 2, base_n_features * 2 )
        if no_blue:
            self.conv12 = BasicConv2d_311(base_n_features * 2, base_n_features * 2 )
        else:
            self.conv12 = Bluestack(base_n_features * 2, base_n_features * 2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv13 = BasicConv2d_311(base_n_features * 2 + base_n_features, base_n_features)
        self.conv14 = BasicConv2d_311(base_n_features * 1, base_n_features // 2 )

        self.conv23 = BasicConv2d_311(base_n_features // 2 + 1, 1 )

        # Skip connnections:
        self.skip1 = Singleconv2d_Ad4(base_n_features, base_n_features)
        self.skip2 = Singleconv2d_Ad4(base_n_features * 2, base_n_features * 2)
        self.skip3 = Singleconv2d_Ad4(base_n_features * 4, base_n_features * 4)
        self.skip4 = Singleconv2d_Ad4(base_n_features * 8, base_n_features * 8)
        self.skip5 = Singleconv2d_Ad4(base_n_features * 16, base_n_features * 16)
        self.skip6 = Singleconv2d_Ad4(base_n_features * 32, base_n_features * 32)
        self.skip7 = Singleconv2d_Ad4(num_input_channels, 1)
    def forward(self, x):
        s7 = self.skip7(x)
        x = self.conv1(x)
        s1 = x = self.conv2(x)
        s1 = self.skip1(s1)
        x = self.down1(x)

        x =  self.conv3(x)
        s2 = x =  self.conv4(x)
        s2 = self.skip2(s2)
        x = self.down2(x)

        x =  self.conv5(x)
        s3 = x =  self.conv6(x)
        s3 = self.skip3(s3)
        x = self.down3(x)

        x =  self.conv7(x)
        s4 = x =  self.conv15(x)
        s4 = self.skip4(s4)
        x = self.down4(x)

        x =  self.conv16(x)
        s5 = x =  self.conv19(x)
        s5 = self.skip5(s5)
        x = self.down5(x)

        s6 = x =  self.conv20(x)
        s6 = self.skip6(s6)
        x =  self.conv21(s6)

        x = self.up5(x)
        x =  self.conv22(torch.cat((x, s5), 1))
        x =  self.conv17(x)

        x = self.up4(x)
        x =  self.conv18(torch.cat((x, s4), 1))
        x =  self.conv8(x)

        x = self.up1(x)
        x =  self.conv9(torch.cat((x, s3), 1))
        x =  self.conv10(x)

        x = self.up2(x)
        x =  self.conv11(torch.cat((x, s2), 1))
        x =  self.conv12(x)

        x = self.up3(x)
        x =  self.conv13(torch.cat((x, s1), 1))
        x =  self.conv14(x)
        x = torch.cat((x, s7), 1)
        x = self.conv23(x)
        return x

djgnet = AsymUNet6layers(1,incep=True,no_blue=True)
# dd = torch.randn(2,1,256,128)
# dd = dd.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)

# from torchsummary import summary
# summary(djgnet,(1,512,128),1)
# from tensorboardX import SummaryWriter
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(djgnet,(dd,))
# WARNING:root:Failed to export an ONNX attribute, since it's not constant, please try to make things (e.g., kernel size) static if possible
# 版本问题