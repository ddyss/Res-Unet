import torch.nn as nn
import torch
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, (1, 1), (1, 1), (0, 0)),#ori code why 110 not 311
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, (1, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, base_ch=64):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=base_ch)
        self.Conv2 = conv_block(ch_in=base_ch, ch_out=base_ch * 2)
        self.Conv3 = conv_block(ch_in=base_ch * 2, ch_out=base_ch * 4)
        self.Conv4 = conv_block(ch_in=base_ch * 4, ch_out=base_ch * 8)
        self.Conv5 = conv_block(ch_in=base_ch * 8, ch_out=base_ch * 16)

        self.ad41 = nn.Conv2d(base_ch, base_ch, (6, 1), (4, 1), (1, 0))
        self.ad42 = nn.Conv2d(base_ch * 2, base_ch * 2, (6, 1), (4, 1), (1, 0))
        self.ad43 = nn.Conv2d(base_ch * 4, base_ch * 4, (6, 1), (4, 1), (1, 0))
        self.ad44 = nn.Conv2d(base_ch * 8, base_ch * 8, (6, 1), (4, 1), (1, 0))
        self.ad4 = nn.Conv2d(base_ch * 16, base_ch * 16, (6, 1), (4, 1), (1, 0))

        self.Up5 = up_conv(ch_in=base_ch * 16, ch_out=base_ch * 8)
        self.Att5 = Attention_block(F_g=base_ch * 8, F_l=base_ch * 8, F_int=base_ch * 4)
        self.Up_conv5 = conv_block(ch_in=base_ch * 16, ch_out=base_ch * 8)

        self.Up4 = up_conv(ch_in=base_ch * 8, ch_out=base_ch * 4)
        self.Att4 = Attention_block(F_g=base_ch * 4, F_l=base_ch * 4, F_int=base_ch * 2)
        self.Up_conv4 = conv_block(ch_in=base_ch * 8, ch_out=base_ch * 4)

        self.Up3 = up_conv(ch_in=base_ch * 4, ch_out=base_ch * 2)
        self.Att3 = Attention_block(F_g=base_ch * 2, F_l=base_ch * 2, F_int=base_ch)
        self.Up_conv3 = conv_block(ch_in=base_ch * 4, ch_out=base_ch * 2)

        self.Up2 = up_conv(ch_in=base_ch * 2, ch_out=base_ch)
        self.Att2 = Attention_block(F_g=base_ch, F_l=base_ch, F_int=32)
        self.Up_conv2 = conv_block(ch_in=base_ch * 2, ch_out=base_ch)

        self.Conv_1x1 = nn.Conv2d(base_ch, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        s1 = self.ad41(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        s2 = self.ad42(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        s3 = self.ad43(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        s4 = self.ad44(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.ad4(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=s4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=s3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=s2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=s1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
djgnet = AttU_Net()
# djgnet = djgnet.to('cuda')
# dd = torch.randn(2,1,512,128)
# dd = dd.cuda()
# yy = djgnet(dd)
# print(yy.shape)
# print(yy[1].shape)
# from torchsummary import summary
# summary(djgnet,(1,base_ch * 2,base_ch * 2),1)