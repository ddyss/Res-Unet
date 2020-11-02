import torch.nn as nn
import torch.nn.functional as F
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
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out
class BasicConv2d_311(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d_311, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class Ad2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Ad2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (4, 3), (2, 1), 1),
            nn.BatchNorm2d(out_ch), #
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class U_cnn(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32):  # 16 #24ist auch gut):
        super(U_cnn, self).__init__()

        self.conv1 = BasicConv2d_311(num_input_channels,base_n_features)
        self.conv2 = Bluestack(base_n_features,base_n_features)
        self.down1 = nn.MaxPool2d(2)

        self.conv3 = Ad2(base_n_features, base_n_features * 2)
        self.conv4 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.down2 = nn.MaxPool2d(2)

        self.conv5 = Ad2(base_n_features*2, base_n_features * 4)
        self.conv6 = Bluestack(base_n_features * 4, base_n_features * 4)

        self.conv7 = BasicConv2d_311(base_n_features * 4, base_n_features * 2)
        self.conv8 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.up1 = nn.Upsample(scale_factor=2)

        self.conv9 = BasicConv2d_311(base_n_features * 2, base_n_features)
        self.conv10 = Bluestack(base_n_features, base_n_features)
        self.up2 = nn.Upsample(scale_factor=2)

        self.conv11 = BasicConv2d_311(base_n_features, 1)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.down1(x)

        x = self.conv4(self.conv3(x))
        x = self.down2(x)

        x = self.conv6(self.conv5(x))

        x = self.conv8(self.conv7(x))
        x = self.up1(x)
        x = self.conv10(self.conv9(x))
        x = self.up2(x)
        x = self.conv11(x)
        return x

djgnet = U_cnn(1)
# dd = torch.randn(2,1,256,128)
# dd = dd.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,512,128),1)

