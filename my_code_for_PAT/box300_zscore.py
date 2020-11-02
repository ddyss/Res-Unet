import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
from PIL import Image
import scipy
from scipy.stats import pearsonr
import math
from nested import djgnet
def psnr(img1, img2):
    # diff = img1 - img2
    diff = img1 / 255. - img2 / 255.#notnorm
    diff = diff.flatten('C')#这样对图片也适用
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
root_model = 'F:/unet_paper_e2e/pre31denseblock/debug6_derive/final_best_real/'
# root_model = 'F:/unet_paper_e2e/pre28_6layers/72658/'
djgnet=torch.load(root_model + 'unet_paper.pkl',map_location='cpu')# djgnet=djgnet.module
djgnet=djgnet.to('cuda')
roottest = 'F:/train/train_dataset/new_machine/test/box/'
rootsave = 'F:/train/train_dataset/new_machine/test/box/resubmit/'
lazy = 'db_unet'
# line = [i for i in range(129, 385)]
# print(line)
# line = np.array(line)
# print(type(line))
def testpre_nested(c):
    c=c.T
    line = []
    for i in range(129, 385):
        line.append(i)
    c = c[line,:]
    c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
    c = zscorenorm(c)
    c=c.astype(np.float32)
    c=torch.tensor(c)
    c=c.reshape(1,1,128,128)
    c=c.to('cuda')
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa
def testpre(c):
    c = c.T
    c = zscorenorm(c)
    c=c.astype(np.float32)
    c=torch.tensor(c)
    c=c.reshape(1,1,512,128)
    c=c.to('cuda')
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa
pc_total = []
psnr_total = []
# ctest = scipy.io.loadmat(roottest + 'test300_70.mat')['djg']
# ctrue = scipy.io.loadmat(roottest + 'test300_true.mat')['djg3'][0][:]
# new300 = map(zscorenorm,ctrue)
# new300 = list(new300)
rootpre14 = 'C:/Users/27896\Desktop\paper\conven_testdata/'
newcc = scipy.io.loadmat(rootpre14 + 'test300_sensor_data_40db')['djg300data'][0][:]
newcc = map(zscorenorm, newcc)
newcc = list(newcc)
scipy.io.savemat(rootpre14 + 'test300_sensor_data_40db_pyzscore.mat',{'test300_sensor_data_40db_pyzscore': newcc})
#经过这个操作的，保存的数据变成三维的了，300*128*512  300*100*100

