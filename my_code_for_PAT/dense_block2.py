import torch
import torch.nn as nn
# Densely connected Residual Blocks (DRERNet)
class _C_block(nn.Module):
    def __init__(self, channel_in, growth):
        super(_C_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=4 * growth, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(4 * growth)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=4 * growth, out_channels=growth, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(growth)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv_1(x)))
        out = self.relu2(self.bn2(self.conv_2(out)))
        return out

class _DR_block2(nn.Module):
    def __init__(self, channel_in, growth):
        super(_DR_block2, self).__init__()

        self.C_Block1 = self.make_layer(_C_block, channel_in, growth)
        self.C_Block2 = self.make_layer(_C_block, channel_in + growth, growth)
        self.C_Block3 = self.make_layer(_C_block, channel_in + 2 * growth, growth)
        self.C_Block4 = self.make_layer(_C_block, channel_in + 3 * growth, growth)
        self.bn1 = nn.BatchNorm2d(channel_in + 4 * growth)
        self.relu1 = nn.PReLU()

    def make_layer(self, block, channel_in, growth):
        layers = []
        layers.append(block(channel_in, growth))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.C_Block1(x)

        conc = torch.cat([x, out], 1)

        out = self.C_Block2(conc)

        conc = torch.cat([conc, out], 1)

        out = self.C_Block3(conc)

        conc = torch.cat([conc, out], 1)

        out = self.C_Block4(conc)

        conc = torch.cat([conc, out], 1)

        out = self.relu1(self.bn1(conc))

        # out = torch.add(out, residual)

        return out

djgnet = _DR_block2(3,8)
# dd = torch.randn(2,3,32,128)
# dd = dd.cuda()
# djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)

# from tensorboardX import SummaryWriter
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(djgnet, (dd, ))