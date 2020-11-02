import torch.nn as nn
import torch.nn.functional as F
import torch
class se_res_block(nn.Module):#这些都是二维的，没有那个两个全连接的东西
    def __init__(self, channel, kernel_size = 3, stride = 1, padding = 1, enable = True):
        super(se_res_block, self).__init__()
        self.enable = enable

        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        if self.enable:
            self.se_conv1 = nn.Conv2d(channel, channel // 16, kernel_size=1)
            self.se_conv2 = nn.Conv2d(channel // 16, channel, kernel_size=1)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))

        if self.enable:
            se = F.avg_pool2d(output, output.size(2))#output.size(2)是512，相当于是取大的，最后pool后结果是batch*channel*1*1
            se = F.relu(self.se_conv1(se))
            se = torch.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output

# dd = torch.randn(2,64,512,128)
# dd = dd.cuda()
# djgnet = se_res_block(64)
# djgnet = djgnet.cuda()
# yy = djgnet(dd)
# print(yy.shape)