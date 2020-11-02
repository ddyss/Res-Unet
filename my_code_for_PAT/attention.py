import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ChannelNet import ChannelNet
# def to_var(x):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x)
class CNN_score(nn.Module):
    def __init__(self, num_channel, hidden_size):
        super(CNN_score, self).__init__()
        self.num_channel = num_channel
        self.hidden_size = hidden_size
        self.layer0 = nn.Conv1d(num_channel, 1, kernel_size=9, padding=4)  # 9 4
        # nn.init.xavier_uniform(self.layer0.weight, gain=nn.init.calculate_gain('tanh'))
        self.layer1 = nn.Conv2d(2, 1, kernel_size=(num_channel, 9), padding=(0, 4), stride=(num_channel, 1))
        # nn.init.xavier_uniform(self.layer1.weight, gain=nn.init.calculate_gain('tanh'))
        self.fc00 = nn.Linear(hidden_size, num_channel)
        # nn.init.xavier_uniform(self.fc00.weight, gain=nn.init.calculate_gain('tanh'))
        self.fc01 = nn.Linear(hidden_size, num_channel)
        # nn.init.xavier_uniform(self.fc01.weight, gain=nn.init.calculate_gain('tanh'))
        self.fc1 = nn.Linear(hidden_size, num_channel)  #
        # nn.init.xavier_uniform(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        self.fc2 = nn.Linear(num_channel, num_channel)  #
        # nn.init.xavier_uniform(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, h_i, pre_h_i, pre_s):
        # shape: batch x channel x hidden_size
        # shape: batch x channel x hidden_size  added by myself
        # shape: batch x channel

        out_h_i = self.layer0(h_i)  # batch x 1 x hidden_size  把channel变成了1
        out_h_i = self.fc00(out_h_i.view(out_h_i.size(0), -1))  # batch x channel
        out_pre_h_i = self.layer0(pre_h_i)  # batch x 1 x something124
        out_pre_h_i = self.fc01(out_pre_h_i.view(out_pre_h_i.size(0), -1))  # batch x channel

        h_i = h_i.view(h_i.size(0), -1, h_i.size(1), h_i.size(2))  # batch x 1 x channel x hidden_size
        pre_h_i = pre_h_i.view(pre_h_i.size(0), -1, pre_h_i.size(1),
                               pre_h_i.size(2))  # batch x 1 x channel x hidden_size
        hh = torch.cat((h_i, pre_h_i), 1)  # batch x 2 x channel x hidden_size

        out_hh = self.layer1(hh)  # batch x 1 x hidden_size 又把channel变成了1
        out_hh = self.fc1(out_hh.view(out_hh.size(0), -1))  # batch x channel
        pre_s = self.fc2(pre_s.cuda())

        out = self.tanh(out_h_i + out_pre_h_i + out_hh + pre_s)
        return out  # batch x channel
class Attention(nn.Module):
    def __init__(self, num_channel, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.attention_layer = CNN_score(num_channel, hidden_size)

    def forward(self, h):
        # shape: batch x seq x channel x hidden_size
        batch_size = h.size(0)
        seq_size = h.size(1)
        num_channel = h.size(2)

        # context_matrix = to_var(
        #     torch.zeros((batch_size, seq_size, num_channel, self.hidden_size)))  # batch x seq x channel x hidden_size
        context_matrix = torch.zeros((batch_size, seq_size, num_channel, self.hidden_size))  # batch x seq x channel x hidden_size
        for i in range(h.size(1)):  # seq_size
            hh_i = h[:, i, :, :]  # current hidden state: batch x channel x hidden_size
            if i == 0:
                # scores = to_var(torch.zeros((batch_size, 1, num_channel)))
                scores = torch.zeros((batch_size, 1, num_channel))
                pre_hh_i = h[:, i, :, :] * 0.0  # batch x channel x hidden_size
            else:
                # scores = to_var(torch.zeros((batch_size, i, num_channel)))  # batch x sub_seq_size x channel
                scores = torch.zeros((batch_size, i, num_channel))  # batch x sub_seq_size x channel
                pre_hh_i = h[:, :i, :, :]  # previous hidden states: batch x sub_seq_size x channel x hidden_size
                for j in range(pre_hh_i.size(1)):  # sub_seq_size
                    if j == 0:
                        # scores[:, j, :] = self.energy(hh_i, pre_hh_i[:, j, :, :],
                        #                               to_var(torch.zeros((batch_size, num_channel))))
                        scores[:, j, :] = self.energy(hh_i, pre_hh_i[:, j, :, :],
                                                      torch.zeros((batch_size, num_channel)))
                    else:
                        pre_score = scores[:, j - 1, :].clone()  # 克隆但不改变本体
                        scores[:, j, :] = self.energy(hh_i, pre_hh_i[:, j, :, :], pre_score)  # batch x channel

            scores = self.normalization(scores, 2)  # batch x sub_seq_size x channel

            scores = scores.view(scores.size(0), scores.size(1), scores.size(2),
                                 -1)  # batch x sub_seq_size x channel x 1
            scores = scores.expand(scores.size(0), scores.size(1), scores.size(2),
                                   self.hidden_size)  # batch x sub_seq_size x channel x hidden_size

            context = pre_hh_i.cuda() * scores.cuda()  # batch x sub_seq_size x channel x hidden_size
            context = context.sum(1)  # batch x channel x hidden_size
            context_matrix[:, i, :, :] = context  # batch x 1 x channel x hidden_size

        # batch x seq x 1 x channel x hidden_size
        # context_matrix = context_matrix.view(context_matrix.size(0), context_matrix.size(1), -1, context_matrix.size(2),
        #                                      context_matrix.size(3))
        # h = h.view(h.size(0), h.size(1), -1, h.size(2), h.size(3))
        out = torch.cat([context_matrix.cuda(), h.cuda()], 1)  # batch x seq x 2 x channel x hidden_size

        return out

    def energy(self, hidden_i, pre_hidden_i, pre_scores):
        # shape: batch x channel x hidden_size
        # shape: batch x channel

        # energies = to_var(torch.zeros((hidden_i.size(0), hidden_i.size(1))))  # batch x channel
        energies = torch.zeros((hidden_i.size(0), hidden_i.size(1)))  # batch x channel
        h_i = hidden_i.clone()
        pre_h_i = pre_hidden_i.contiguous()
        energies = self.attention_layer(h_i, pre_h_i, pre_scores)  # 放CNN_score中

        return energies  # batch x channel

    def normalization(self, scores, gamma):
        # shape: batch x sub_seq_size x channel
        sub_seq_size = scores.size(1)
        num_channel = scores.size(2)
        gamma_d = self.relu(scores).sum(2)  # batch x sub_seq_size
        gamma_d_sum = gamma_d.sum(1, keepdim=True) + 1e-8  # batch x 1
        gamma_d_sum = gamma_d_sum.expand(gamma_d_sum.size(0), sub_seq_size)  # batch x sub_seq_size
        gamma_d = gamma_d / gamma_d_sum  # batch x sub_seq_size
        gamma_d = gamma_d.view(gamma_d.size(0), gamma_d.size(1), -1)  # batch x sub_seq_size x 1
        gamma_d = gamma_d.expand(gamma_d.size(0), gamma_d.size(1), num_channel)  # batch x sub_seq_size x channel

        scores = self.sigmoid(scores)
        # out = to_var(torch.zeros(scores.size(0), sub_seq_size, num_channel))  # batch x sub_seq_size x channel
        out = torch.zeros(scores.size(0), sub_seq_size, num_channel)  # batch x sub_seq_size x channel
        for i in range(sub_seq_size):
            scores_i_sum = scores[:, i, :].sum(1, keepdim=True) + 1e-8  # batch x 1
            scores_i_sum = scores_i_sum.expand(scores_i_sum.size(0), num_channel)  # batch x channel
            out[:, i, :] = scores[:, i, :] / scores_i_sum

        out = gamma_d * out  # batch x sub_seq_size x channel
        out = out.view(out.size(0), -1)  # batch x (sub_seq_size * channel)
        out = F.softmax(gamma * out)  # batch x (sub_seq_size * channel)
        out = out.view(out.size(0), sub_seq_size, num_channel)  # batch x sub_seq_size x channel

        return out
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
class Bluestack(nn.Module):#must  in_ch = out_ch
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
        # self.dropout = nn.Dropout2d(0.5)
    def forward(self, x):
        out = x + self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        # out = self.dropout(out)
        return out
class Singleconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Singleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True, bidirectional=True)
    def forward(self, x, hidden = None):
        out,hidden = self.gru(x, hidden)
        return out
class AsymUNet(nn.Module):
    def __init__(self, num_input_channels, base_n_features=32):  # 16 #24ist auch gut):
        super(AsymUNet, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, base_n_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_n_features)
        self.conv2 = Bluestack(base_n_features, base_n_features)
        self.bn2 = nn.BatchNorm2d(base_n_features)
        self.down1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(base_n_features, base_n_features * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_n_features * 2)
        self.conv4 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.bn4 = nn.BatchNorm2d(base_n_features * 2)
        self.down2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(base_n_features * 2, base_n_features * 4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_n_features * 4)
        self.conv6 = Bluestack(base_n_features * 4, base_n_features * 4)
        self.bn6 = nn.BatchNorm2d(base_n_features * 4)
        self.down3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(base_n_features * 4, base_n_features * 8, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(base_n_features * 8)
        self.conv8 = Bluestack(base_n_features * 8, base_n_features * 8)
        self.bn8 = nn.BatchNorm2d(base_n_features * 8)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv9 = nn.Conv2d(base_n_features * 8 + base_n_features * 4, base_n_features * 4, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(base_n_features * 4)
        self.conv10 = Bluestack(base_n_features * 4, base_n_features * 4)
        self.bn10 = nn.BatchNorm2d(base_n_features * 4)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv11 = nn.Conv2d(base_n_features * 4 + base_n_features * 2, base_n_features * 2, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(base_n_features * 2)
        self.conv12 = Bluestack(base_n_features * 2, base_n_features * 2)
        self.bn12 = nn.BatchNorm2d(base_n_features * 2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv13 = nn.Conv2d(base_n_features * 2 + base_n_features, base_n_features, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(base_n_features)
        self.conv14 = nn.Conv2d(base_n_features * 1, base_n_features//2, 3, padding=1)

        # self.conv15 = nn.Conv2d(1 + 1, 1, 3, 1, 1)
        self.conv15 = nn.Conv2d(base_n_features//2 + 1, 1, 3, 1, 1)
        # self.conv15 = Narrow(base_n_features // 2, 1)

        # Skip connnections:
        self.skip1 = nn.Conv2d(base_n_features, base_n_features, (6, 3), (4, 1), (1, 1))
        self.skip2 = nn.Conv2d(base_n_features * 2, base_n_features * 2, (6, 3), (4, 1), (1, 1))
        self.skip3 = nn.Conv2d(base_n_features * 4, base_n_features * 4, (6, 3), (4, 1), (1, 1))
        self.skip4 = nn.Conv2d(base_n_features * 8, base_n_features * 8, (6, 3), (4, 1), (1, 1))
        self.skip5 = nn.Conv2d(1, 1, (6, 3), (4, 1), (1, 1))

    def forward(self, x):
        s5 = self.skip5(x)
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
        x = torch.cat((x, s5), 1)
        x = self.conv15(x)
        return x
class TotalNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TotalNet,self).__init__()
        self.conv1 = CNN_0()#out: batch x 32 x 128 x 128   all
        self.convn1 = AsymUNet(1)#out: batch x 1 x 128 x 128  all
        self.conv2 = ChannelNet(128)#out: batch x 32 x 128 x 128  local
        self.gru = RNN(128, 64, 2)
        self.att = Attention(128,128)#out: batch x 64 x 128 x 128
        self.conv3 = Singleconv(in_ch,out_ch)
        self.conv4 = Singleconv(out_ch, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = x2.cuda()
        x3 = torch.zeros(x2.size(0),x2.size(1),x2.size(2),x2.size(3))
        x3 = x3.cuda()
        for i in range(x2.size(3)):
            x3[:,:,:,i] = self.gru(x2[:,:,:,i])
        x3 = self.att(x3)
        x = torch.cat([x1,x3],1)
        out = self.conv4(self.conv3(x))
        return out
dd = torch.randn(2,1,512,128)
# dd = torch.randn(2,1,128)
# dd = torch.randn(2,32,128,128)
dd = dd.cuda()
# djgnet = TotalNet(33,16)
# djgnet = TotalNet(34,16)
# djgnet = TotalNet(64,32)
djgnet = TotalNet(96,32)
# djgnet = RNN(128,64,2)
# djgnet = Attention(128,128)
djgnet = djgnet.to('cuda')
# yy = djgnet(dd)
# print(yy.shape)

# from torchsummary import summary
# summary(djgnet,(1,512,128),1)