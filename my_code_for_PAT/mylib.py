import torch
import torch.nn as nn
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class CNN_0(nn.Module):
    def __init__(self):
        super(CNN_0, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(1, 8, 5, 1, 2)
        self.conv1_3 = nn.Conv2d(1, 8, 7, 1, 3)
        self.conv1_4 = nn.Conv2d(1, 8, 9, 1, 4)

        self.norm1 = nn.BatchNorm2d(32)  #
        self.pool1 = nn.AvgPool2d((4, 1), stride=(2, 1), padding=(1, 0))  # 5

        self.conv2_1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(32, 16, 5, 1, 2)

        self.norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((4, 1), stride=(2, 1), padding=(1, 0))  #

    def forward(self, x):        # shape: batch x
        encoded = torch.cat([self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)], 1)
        encoded = self.relu(self.norm1(encoded))
        encoded = self.pool1(encoded)

        encoded = torch.cat([self.conv2_1(encoded), self.conv2_2(encoded)], 1)
        encoded = self.relu(self.norm2(encoded))
        encoded = self.pool2(encoded)
        return encoded
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Expand(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Expand,self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, (21,3),1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class Narrow(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Narrow,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 31,1,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
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
class resblock_cat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(resblock_cat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.bn = nn.BatchNorm2d(in_ch + out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = torch.cat([x,self.conv(x)],1)
        out = self.leaky_relu(self.bn(x))
        return out
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
class ABd2(nn.Module):#输出channel=2*out_ch
    def __init__(self, in_ch, out_ch):
        super(ABd2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch), #
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        # self.bn = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = torch.cat([self.conv(x),self.skip(x)],1)
        return out
class Unet(nn.Module):
    def __init__(self,in_ch):
        super(Unet,self).__init__()
        self.doubleconv1 = DoubleConv(in_ch,in_ch*4)
        self.down = nn.MaxPool2d(2)
        self.constant = Constant(in_ch*4,in_ch*4)
        self.up = nn.Upsample(scale_factor=2)
        self.doubleconv2 = DoubleConv(in_ch*4,in_ch)
    def forward(self, x):
        sl = self.doubleconv1(x)
        x = self.down(sl)
        x = self.constant(x)
        sr = self.up(x)
        x = sl + sr
        x = self.doubleconv2(x)
        return x
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
class Multi_kernel(nn.Module):
    def __init__(self):
        super(Multi_kernel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(1, 8, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(1, 8, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(1, 8, 5, 1, 2)
        self.conv1_4 = nn.Conv2d(1, 8, 7, 1, 3)

        self.norm1 = nn.BatchNorm2d(32)

        # self.convf1 = Singleconv(32,16)
        # self.convf2 = Singleconv(16,1)

    def forward(self, x):        # shape: batch x
        encoded = torch.cat([self.conv1_1(x), self.conv1_2(x), self.conv1_3(x), self.conv1_4(x)], 1)
        encoded = self.relu(self.norm1(encoded))
        # encoded = self.pool1(encoded)

        # encoded = torch.cat([self.conv2_1(encoded), self.conv2_2(encoded)], 1)
        # encoded = self.relu(self.norm2(encoded))
        # encoded = self.pool2(encoded)
        #
        # encoded = self.convf2(self.convf1(encoded))
        return encoded
class Inceptionbegin(nn.Module):

    def __init__(self, in_channels, branch_channels):
        super(Inceptionbegin, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, branch_channels, kernel_size = 1, stride = 1, padding = 0)
        self.branch3x3 = BasicConv2d(in_channels, branch_channels, kernel_size = 3, stride = 1, padding = 1)
        self.branch5x5 = BasicConv2d(in_channels, branch_channels, kernel_size = 5, stride = 1, padding = 2)
        self.branch7x7 = BasicConv2d(in_channels, branch_channels, kernel_size = 7, stride = 1, padding = 3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch7x7]
        return torch.cat(outputs, 1)

djgnet = resblock_cat(1,16)
# dd = torch.randn(2,1,512,128)
# dd = dd.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,512,128),1)


