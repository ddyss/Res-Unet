import matplotlib.pyplot as plt
import scipy.io
import torch
import math
import numpy as np
import cv2
from scipy.stats import pearsonr
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
def psnr(img1, img2):
    diff = img1 - img2
    # diff = img1 / 255. - img2 / 255.#notnorm
    diff = diff.flatten('C')#这样对图片也适用
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
pathraw = 'F:/Study_CuSO4/Scan_1/deal_t1/'
# pathraw = 'F:/Study_iRFP713/Scan_1/deal_t1/'
# pathraw = 'F:/Study_119djg/Scan_2/deal_t1/'
# pathraw_bprecon = pathraw + 'RECONs/'
root_model = 'F:/unet_paper_e2e/pre28_6layers/72658/'
# path_save = 'F:/Study_119djg/DL/'
savename = 'del_threshold_smooth'
djgnet=torch.load(root_model + 'unet_paper.pkl',map_location='cpu')
djgnet = djgnet.cuda()
def testpre(c):
    # c = c.T
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
def testpre_nested(c):
    c=c.T
    line = []
    for i in range(129, 385):
        line.append(i)
    c = c[line, :]
    c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
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

# lazy = 'dere'
# GROUND_TRUTH = scipy.io.loadmat(pathraw_bprecon + 't1.mat')['t1']
# GROUND_TRUTH = cv2.resize(GROUND_TRUTH,(128,128),interpolation=cv2.INTER_NEAREST)
# GROUND_TRUTH = zscorenorm(GROUND_TRUTH)
# num11 = GROUND_TRUTH.flatten()

# bp
# root_bprecon = 'F:/train/train_dataset/new_machine/test/bp_collection/'
# c = scipy.io.loadmat(root_bprecon + lazy + '_sensor_data_100_100_bp.mat')['back2']
# aaa = cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
# aaa = zscorenorm(aaa)
# num12 = aaa.flatten()

c = scipy.io.loadmat(pathraw + savename + '.mat')['smoo']
# c[0:899] = c[1130:2029]#前400行用后400行替换
c = cv2.resize(c,(128,512),interpolation=cv2.INTER_NEAREST)
aaa = testpre(c)
# num12 = aaa.flatten()

# p1 = pearsonr(num11,num12)[0]
# p1 = round(p1,2)
# p2 = round(psnr(num11,num12),2)
# print(p1,p2)

fig, ax = plt.subplots(nrows=1, figsize=(6,6))
im = ax.imshow(aaa, extent=[0, 1, 0, 1])
# ax.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
cb=plt.colorbar(im)
# plt.savefig(path_save + savename + '.png')
plt.show()
