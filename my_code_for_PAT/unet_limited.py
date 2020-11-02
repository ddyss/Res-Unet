import torch.nn as nn
import torch
import torch.nn.functional as F
class Constant(nn.Module):#输出channel=in_ch
    def __init__(self, in_ch, out_ch):
        super(Constant, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = x + self.conv(x)
        return out
class DoubleConv(nn.Module):#输出channel=out_ch
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
class Unet_Limited(nn.Module):
    def __init__(self,in_ch,medium_ch = 16):
        super(Unet_Limited,self).__init__()
        self.constant = Constant(medium_ch * 8,medium_ch * 8)
        self.conv1 = DoubleConv(in_ch,medium_ch)
        self.down1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(medium_ch,medium_ch*2)
        self.down2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(medium_ch * 2, medium_ch * 4)
        self.down3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(medium_ch * 4, medium_ch * 8)
        self.down4 = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv5 = DoubleConv(medium_ch*8,medium_ch*4)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(medium_ch * 4, medium_ch*2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.conv7 = DoubleConv(medium_ch * 2, medium_ch)
        self.up4 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(medium_ch, 1)
    def forward(self, x):
        s1 = self.conv1(x)
        x = self.down1(s1)
        s2 = self.conv2(x)
        x = self.down2(s2)
        s3 = self.conv3(x)
        x = self.down3(s3)
        s4 = self.conv4(x)
        x = self.down4(s4)
        x = self.constant(x)
        sr1 = self.up1(x)
        x = s4 + sr1
        x = self.conv5(x)
        sr2 = self.up2(x)
        x = s3 + sr2
        x = self.conv6(x)
        sr3 = self.up3(x)
        x = s2 + sr3
        x = self.conv7(x)
        sr4 = self.up4(x)
        x = s1 + sr4
        x = self.conv8(x)
        return x
class AsymUNet(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32):  # 16 #24ist auch gut):
        super(AsymUNet, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, base_n_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_n_features)
        self.conv2 = nn.Conv2d(base_n_features, base_n_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_n_features)
        self.down1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(base_n_features, base_n_features * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_n_features * 2)
        self.conv4 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_n_features * 2)
        self.down2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(base_n_features * 2, base_n_features * 4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_n_features * 4)
        self.conv6 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(base_n_features * 4)
        self.down3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(base_n_features * 4, base_n_features * 8, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(base_n_features * 8)
        self.conv8 = nn.Conv2d(base_n_features * 8, base_n_features * 8, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(base_n_features * 8)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv9 = nn.Conv2d(base_n_features * 8 + base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(base_n_features * 4)
        self.conv10 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv11 = nn.Conv2d(base_n_features * 4 + base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(base_n_features * 2)
        self.conv12 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(base_n_features * 2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv13 = nn.Conv2d(base_n_features * 2 + base_n_features, base_n_features, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(base_n_features)
        self.conv14 = nn.Conv2d(base_n_features * 1, 1, 3, padding=1)

        # Skip connnections:
        self.skip1 = nn.Conv2d(base_n_features, base_n_features, 3,1,1)
        self.skip2 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3,1,1)
        self.skip3 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3,1,1)
        self.skip4 = nn.Conv2d(base_n_features * 8, base_n_features * 8, 3,1,1)

    def forward(self, x):
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

        s4 = x = F.leaky_relu(self.bn7(self.conv7(x)))
        s4 = self.skip4(s4)
        x = F.leaky_relu(self.bn8(self.conv8(s4)))

        x = self.up1(x)
        x = F.leaky_relu(self.bn9(self.conv9(torch.cat((x, s3), 1))))
        x = F.leaky_relu(self.bn10(self.conv10(x)))

        x = self.up2(x)
        x = F.leaky_relu(self.bn11(self.conv11(torch.cat((x, s2), 1))))
        x = F.leaky_relu(self.bn12(self.conv12(x)))

        x = self.up3(x)
        x = F.leaky_relu(self.bn13(self.conv13(torch.cat((x, s1), 1))))
        x = F.leaky_relu(self.conv14(x))

        return x
# dd = torch.randn(4,1,128,512)
# dd = dd.cuda()
# djgnet = AsymUNet(1)
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)