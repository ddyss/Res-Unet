import torch
import torch.nn as nn
from torch.nn import init
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+ 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)
class UNet_Nested(nn.Module):  #final 7 太过庞大了，尚未设置

    def __init__(self, in_channels=1, n_classes=1, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        # filters = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # filters = [32, 64, 128, 256, 512, 1024, 2048]
        filters = [64, 128, 256, 512, 1024, 2048, 4096]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.conv50 = unetConv2(filters[4], filters[5], self.is_batchnorm)
        self.conv60 = unetConv2(filters[5], filters[6], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat41 = unetUp(filters[5], filters[4], self.is_deconv)
        self.up_concat51 = unetUp(filters[6], filters[5], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat32 = unetUp(filters[4], filters[3], self.is_deconv, 3)
        self.up_concat42 = unetUp(filters[5], filters[4], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat23 = unetUp(filters[3], filters[2], self.is_deconv, 4)
        self.up_concat33 = unetUp(filters[4], filters[3], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.up_concat14 = unetUp(filters[2], filters[1], self.is_deconv, 5)
        self.up_concat24 = unetUp(filters[3], filters[2], self.is_deconv, 5)

        self.up_concat05 = unetUp(filters[1], filters[0], self.is_deconv, 6)
        self.up_concat15 = unetUp(filters[2], filters[1], self.is_deconv, 6)

        self.up_concat06 = unetUp(filters[1], filters[0], self.is_deconv, 7)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_5 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_6 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpool(X_00)    # 16*256*256
        X_10= self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        maxpool4 = self.maxpool(X_40)
        X_50 = self.conv50(maxpool4)
        maxpool5 = self.maxpool(X_50)
        X_60 = self.conv60(maxpool5)
        # column : 1
        X_01 = self.up_concat01(X_10 ,X_00)
        X_11 = self.up_concat11(X_20 ,X_10)
        X_21 = self.up_concat21(X_30 ,X_20)
        X_31 = self.up_concat31(X_40 ,X_30)
        X_41 = self.up_concat41(X_50, X_40)
        X_51 = self.up_concat51(X_60, X_50)
        # column : 2
        X_02 = self.up_concat02(X_11 ,X_00 ,X_01)
        X_12 = self.up_concat12(X_21 ,X_10 ,X_11)
        X_22 = self.up_concat22(X_31 ,X_20 ,X_21)
        X_32 = self.up_concat32(X_41, X_30, X_31)
        X_42 = self.up_concat42(X_51, X_40, X_41)
        # column : 3
        X_03 = self.up_concat03(X_12 ,X_00 ,X_01 ,X_02)
        X_13 = self.up_concat13(X_22 ,X_10 ,X_11 ,X_12)
        X_23 = self.up_concat23(X_32, X_20, X_21, X_22)
        X_33 = self.up_concat33(X_42, X_30, X_31, X_32)
        # column : 4
        X_04 = self.up_concat04(X_13 ,X_00 ,X_01 ,X_02 ,X_03)
        X_14 = self.up_concat14(X_23, X_10, X_11, X_12, X_13)
        X_24 = self.up_concat24(X_33, X_20, X_21, X_22, X_23)
        # column : 5
        X_05 = self.up_concat05(X_14, X_00, X_01, X_02, X_03, X_04)
        X_15 = self.up_concat15(X_24, X_10, X_11, X_12, X_13, X_14)
        # column : 6
        X_06 = self.up_concat06(X_15, X_00, X_01, X_02, X_03, X_04,X_05)
        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final_5 = self.final_4(X_05)
        final_6 = self.final_5(X_06)

        final = (final_1 + final_2 + final_3 + final_4 + final_5 + final_6) / 6
        # final = (final_1 + final_2 + final_3 + final_4 + final_5) / 5
        # final = (final_1 +final_2 +final_3 +final_4 ) /4
        # final = (final_1 + final_2 + final_3) / 3

        if self.is_ds:
            return final
        else:
            return final_6

dd = torch.randn(2,1,128,128)
dd = dd.cuda()
djgnet = UNet_Nested()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)
# from torchsummary import summary
# summary(djgnet,(1,128,128),1)
# from tensorboardX import SummaryWriter
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(djgnet,(dd,))