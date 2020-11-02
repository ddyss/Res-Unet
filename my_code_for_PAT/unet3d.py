import torch
import torch.nn as nn
import torch.nn.functional as F
# '''
class Unet3D(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32):  # 16 #24ist auch gut):
        super(Unet3D, self).__init__()

        self.conv1 = nn.Conv3d(num_input_channels, base_n_features, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(base_n_features)
        self.conv2 = nn.Conv3d(base_n_features, base_n_features, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(base_n_features)
        self.down1 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(base_n_features, base_n_features * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(base_n_features * 2)
        self.conv4 = nn.Conv3d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(base_n_features * 2)
        self.down2 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(base_n_features * 2, base_n_features * 4, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(base_n_features * 4)
        self.conv6 = nn.Conv3d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn6 = nn.BatchNorm3d(base_n_features * 4)
        self.down3 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(base_n_features * 4, base_n_features * 8, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(base_n_features * 8)
        self.conv8 = nn.Conv3d(base_n_features * 8, base_n_features * 8, 3, padding=1)
        self.bn8 = nn.BatchNorm3d(base_n_features * 8)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv9 = nn.Conv3d(base_n_features * 8 + base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn9 = nn.BatchNorm3d(base_n_features * 4)
        self.conv10 = nn.Conv3d(base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn10 = nn.BatchNorm3d(base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv11 = nn.Conv3d(base_n_features * 4 + base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn11 = nn.BatchNorm3d(base_n_features * 2)
        self.conv12 = nn.Conv3d(base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn12 = nn.BatchNorm3d(base_n_features * 2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv13 = nn.Conv3d(base_n_features * 2 + base_n_features, base_n_features, 3, padding=1)
        self.bn13 = nn.BatchNorm3d(base_n_features)
        self.conv14 = nn.Conv3d(base_n_features * 1, 1, 3, padding=1)

        # Skip connnections:
        self.skip1 = nn.Conv3d(base_n_features, base_n_features, (3, 3, 4), (1, 1, 2), (1, 1, 1))
        self.skip2 = nn.Conv3d(base_n_features * 2, base_n_features * 2, (3, 3, 4), (1, 1, 2), (1, 1, 1))
        self.skip3 = nn.Conv3d(base_n_features * 4, base_n_features * 4, (3, 3, 4), (1, 1, 2), (1, 1, 1))
        self.skip4 = nn.Conv3d(base_n_features * 8, base_n_features * 8, (3, 3, 4), (1, 1, 2), (1, 1, 1))

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
        x = torch.cat((x, s1), 1)
        x = F.leaky_relu(self.bn13(self.conv13(x)))
        x = F.leaky_relu(self.conv14(x))

        return x
# '''
dd = torch.randn(2,1,32,32,128)
dd = dd.cuda()
djgnet = Unet3D(1)
djgnet = djgnet.to('cuda')
yy = djgnet(dd)
print(yy.shape)
from torchsummary import summary
summary(djgnet,(1,32,32,128),1)
