import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import h5py
import cv2

# '''
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# root_model = 'F:/unet_paper_e2e/pre28_6layers/resunet_6layers_notop/'
# a=np.load(root_model + 'trainloss.npy')
# plt.plot(a,'r-',lw=5)
# a2=np.load(root_model + 'validloss.npy')
# plt.plot(a2,'b-',lw=5)
# plt.title('r-train,b-val')
# # a50 = a2[0:50]
# posi = np.where(a2 == np.min(a2))
# print(len(a2))
# print(np.min(a2))
# print(posi)
# # plt.savefig(root_model + 'loss.png')
# plt.show()
# '''  sensor_data
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
def max_min(x):
    aa = np.max(x)
    bb = np.min(x)
    new = (x-bb)/(aa-bb)
    return new
# rootpre14 = 'C:/Users/27896/Desktop/paper/conven_testdata/BP_denoise/'
lazy = 'dere'
# aaa = scipy.io.loadmat(rootpre14 + 'pre_' + lazy + '_bp.mat')['pre_' + lazy + '_bp']
# rootpre14 = 'F:/Study_CuSO4/Scan_1/'
# aaa = scipy.io.loadmat(rootpre14 + 't1.mat')['t1']
root_save = 'C:/Users/27896/Desktop/paper/conven_testdata/'
aaa = scipy.io.loadmat(root_save + lazy + '_sensor_data.mat')['sensor_data_40db']
# aaa = cv2.resize(aaa,(128,128),interpolation=cv2.INTER_NEAREST)
# aaa = zscorenorm(aaa)
aaa = aaa.T
scipy.io.savemat(root_save + 'mrr_recon.mat', {'mrr_recon': aaa})
fig, ax = plt.subplots(nrows=1)#figsize=(6,6)是设置子图长宽（600，600)
im = ax.imshow(aaa, extent=[0, 1, 0, 1],vmin=0, vmax=0.007)# 8 4 10 2.5            ,vmin=0, vmax=4
position=fig.add_axes([0.85, 0.10, 0.03, 0.7])#left, bottom, width, height :bar距离左边，距离底部，bar的宽度，bar的高度
cb=plt.colorbar(im,cax=position,orientation='vertical')#'horizontal'
plt.show()

#计算flops，下面两种方法计算结果一样
# import torch
# from asym_6layers_best import djgnet
# from thop import profile
# # from thop import clever_format
# model = djgnet
# input = torch.randn(1, 1, 512, 128)
# # input = torch.randn(1, 1, 128, 128)
# macs, params = profile(model, inputs=(input, ))
# print(macs,params)
# # macs, params = clever_format([macs, params], "%.3f")#保留3位有效数字，最后一位四舍五入
# # print(macs,params)

# from asym import djgnet
# import torch
# from ptflops import get_model_complexity_info
#
# with torch.cuda.device(0):
#   macs, params = get_model_complexity_info(djgnet, (1, 512, 128), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#### 像mesh函数那样3D显示
#### plt.subplot(236)
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# size=aaa.shape
# Y=np.arange(0,size[0],1)
# X=np.arange(0,size[1],1)
# X,Y=np.meshgrid(X,Y)
# fig=plt.figure()
# # ax=fig.gca(projection='3d')#0.99版本
# ax = Axes3D(fig)#1.版本
# ax.plot_surface(X,Y,aaa,cmap='rainbow')
# plt.show()

# aaa=cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST)
## INTER_NEAREST最近邻插值法， 找到与之距离最相近的邻居（原来就存在的像素点， 黑点）， 赋值与其相同。用这个
## INTER_LINEAR线性插值（默认），根据四周取平均
## INTER_CUBIC由相邻的4*4像素计算得出，公式类似于双线性.
## INTER_LANCZOS4 Lanczos插值 由相邻的8*8像素计算得出，公式类似于双线性
# print(np.max(aaa))
# print(np.min(aaa))
# sum(-1<i<2 for i in c[1][:]) #只适合一列list，列出大于1的个数

# scipy.io.savemat('pre11/pree90r31_70.mat',{'testPredict': aaa})

##查看cell里单个的数组矩阵
# f = h5py.File('F:/train/train_dataset/s_rand_30_12_30/s_rand_30_360_true.mat','r')
# print(aaa.keys())   #可以查看读取数据（.mat）里面的名字
# data = f.get('djg2')  #.value
# print(data)          #查看mat的形状
# test = f['djg2']
# st = test[0][10]
# obj = f[st]
# aaa=np.array(obj)
# print(aaa.shape)

# from torchsummary import summary
# summary(djgnet,(1,128,512),10)

# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
# print(get_parameter_number(djgnet))#该函数只查看参数个数，相当于sunmmary的一个小功能
# print(djgnet.state_dict())#查看每一个参数的大小

# from tensorboardX import SummaryWriter
# model = djgnet(1)
# print("总参数量:",model.count_params())

# 三种可视化方式------这些显示都不能让unet弯过来
# from tensorboardX import SummaryWriter
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(djgnet,(dd,))  #tensorboard --logdir Jul22_18-03-19_WH-PC19012AlexNet  打开tensorboard控制台

# import tensorwatch as tw #装是装上了，居然报错，搜索显示是Torch版本问题
# # model=djgnet()
# tw.draw_model(djgnet, [2, 1, 512, 128])

# from torchviz import make_dot  #之前还单独安装了一个Git，graphviz和torchviz都得单独安装，而不是pip
# g = make_dot(yy)
# g.render('net_model', view=True)

# import torchvision
# torchvision.models.alexnet()
# torchvision.models.vgg11()
# torchvision.models.googlenet()
# torchvision.models.inception_v3()
# torchvision.models.resnet18()
# torchvision.models.resnext50_32x4d()
# torchvision.models.mobilenet_v2()
# torchvision.models.densenet121()
# torchvision.models.segmentation.deeplabv3_resnet50()

# djgnet.load_state_dict(torch.load('F:/unet_paper_e2e/pre21nested/unet_paper_params.pkl'))#前面不能有model=
# djgnet=torch.load('F:/unet_paper_e2e/pre19/6层/addblue_noaug3/unet_paper.pkl',map_location='cpu')

# from  PIL import Image
# root='C:/Users/djg/Pictures/pat/pat.jpg'
# pic = Image.open(root)
# pic = pic.resize((220, 220))

# from skimage import transform,data
# dst=transform.resize(img, (80, 60))

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True      #程序按需申请内存
# sess = tf.Session(config = config)

# scipy.io.savemat('/data1/MIP1/dataset/t0.mat',{'ground': twe_ori[0]})
# scipy.io.savemat('/data1/MIP1/dataset/t1.mat',{'ground': twe_ori[1]})
# scipy.io.savemat('/data1/MIP1/dataset/t2.mat',{'ground': twe_ori[2]})
# scipy.io.savemat('/data1/MIP1/dataset/t3.mat',{'ground': twe_ori[3]})
# scipy.io.savemat('/data1/MIP1/dataset/t4.mat',{'ground': twe_ori[4]})
# scipy.io.savemat('/data1/MIP1/dataset/t5.mat',{'ground': twe_ori[5]})
# scipy.io.savemat('/data1/MIP1/dataset/t6.mat',{'ground': twe_ori[6]})
# scipy.io.savemat('/data1/MIP1/dataset/t7.mat',{'ground': twe_ori[7]})
# scipy.io.savemat('/data1/MIP1/dataset/t8.mat',{'ground': twe_ori[8]})
# scipy.io.savemat('/data1/MIP1/dataset/t9.mat',{'ground': twe_ori[9]})

'''
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
        self.skip1 = nn.Conv2d(base_n_features, base_n_features, (20, 3), (20, 1), (9, 1))
        self.skip2 = nn.Conv2d(base_n_features * 2, base_n_features * 2, (20, 3), (20, 1), (9, 1))
        self.skip3 = nn.Conv2d(base_n_features * 4, base_n_features * 4, (20, 3), (20, 1), (9, 1))
        self.skip4 = nn.Conv2d(base_n_features * 8, base_n_features * 8, (20, 3), (20, 1), (9, 1))

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
'''
