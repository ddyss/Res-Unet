import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#这是分割里network中的att-unet，加一个resize卷积来做端到端
#这个权重只是在这定义了一下，只出现了一次，class中没有用到，可能后面有吧

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

#输出是F_l通道数，第二格
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, num_input_channels = 1, base_n_features=32):  # 16 #24ist auch gut):
        super(AttU_Net, self).__init__()

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

        # self.Att3 = Attention_block(F_g=base_n_features * 4, F_l=base_n_features * 8, F_int=base_n_features * 4)
        # self.Att2 = Attention_block(F_g=base_n_features * 2, F_l=base_n_features * 4, F_int=base_n_features * 2)
        # self.Att1 = Attention_block(F_g=base_n_features * 1, F_l=base_n_features * 2, F_int=base_n_features * 1)

        self.Att3 = Attention_block(F_g=base_n_features * 8, F_l=base_n_features * 4, F_int=base_n_features * 4)
        self.Att2 = Attention_block(F_g=base_n_features * 4, F_l=base_n_features * 2, F_int=base_n_features * 2)
        self.Att1 = Attention_block(F_g=base_n_features * 2, F_l=base_n_features * 1, F_int=base_n_features * 1)

        self.up3 = nn.Upsample(scale_factor=2)
        # self.conv9 = nn.Conv2d(base_n_features * 8 + base_n_features * 8, base_n_features * 4, 3, padding=1)
        self.conv9 = nn.Conv2d(base_n_features * 8 + base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(base_n_features * 4)
        self.conv10 = nn.Conv2d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        # self.conv11 = nn.Conv2d(base_n_features * 4 + base_n_features * 4, base_n_features * 2, 3, padding=1)
        self.conv11 = nn.Conv2d(base_n_features * 4 + base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(base_n_features * 2)
        self.conv12 = nn.Conv2d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(base_n_features * 2)

        self.up1 = nn.Upsample(scale_factor=2)
        # self.conv13 = nn.Conv2d(base_n_features * 2 + base_n_features * 2, base_n_features, 3, padding=1)
        self.conv13 = nn.Conv2d(base_n_features * 2 + base_n_features * 1, base_n_features, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(base_n_features)
        self.conv14 = nn.Conv2d(base_n_features * 1, 1, 3, padding=1)

        # Skip connnections:
        self.skip1 = nn.Conv2d(base_n_features, base_n_features, (6, 3), (4, 1), (1, 1))
        self.skip2 = nn.Conv2d(base_n_features * 2, base_n_features * 2, (6, 3), (4, 1), (1, 1))
        self.skip3 = nn.Conv2d(base_n_features * 4, base_n_features * 4, (6, 3), (4, 1), (1, 1))
        self.skip4 = nn.Conv2d(base_n_features * 8, base_n_features * 8, (6, 3), (4, 1), (1, 1))

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

        up3 = self.up3(x)
        # x = self.Att3(s3,up3)
        x = self.Att3(up3, s3)
        x = F.leaky_relu(self.bn9(self.conv9(torch.cat((x, up3), 1))))
        x = F.leaky_relu(self.bn10(self.conv10(x)))

        up2 = self.up2(x)
        # x = self.Att2(s2, up2)
        x = self.Att2(up2, s2)
        x = F.leaky_relu(self.bn11(self.conv11(torch.cat((x, up2), 1))))
        x = F.leaky_relu(self.bn12(self.conv12(x)))

        up1 = self.up1(x)
        # x = self.Att1(s1, up1)
        x = self.Att1(up1, s1)
        x = F.leaky_relu(self.bn13(self.conv13(torch.cat((x, up1), 1))))
        x = F.leaky_relu(self.conv14(x))

        return x

djgnet = AttU_Net()
# dd = torch.randn(2,1,512,128)
# dd = dd.cuda()
# # dd2 = torch.randn(2,256,16,16)
# # dd2 = dd2.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# # print(yy[1].shape)