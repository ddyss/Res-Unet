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
root_raw = 'F:/train/train_dataset/new_machine/test/'
root_bp = 'F:/train/train_dataset/new_machine/test/bp_collection/'
root_gt = root_raw
root_model = 'F:/unet_paper_e2e/pre35ynet/'
# ceshi zhenshi shuju
djgnet=torch.load(root_model + 'unet_paper.pkl',map_location='cpu')
djgnet = djgnet.cuda()
def testpre(c,bp):
    c = c.T
    c = zscorenorm(c)
    c=c.astype(np.float32)
    c=torch.tensor(c)
    c=c.reshape(1,1,512,128)
    c=c.to('cuda')

    bp = bp.T
    bp = cv2.resize(bp, (128, 128), interpolation=cv2.INTER_NEAREST)
    bp = zscorenorm(bp)
    bp = bp.astype(np.float32)
    bp = torch.tensor(bp)
    bp = bp.reshape(1, 1, 128, 128)
    bp = bp.to('cuda')
    with torch.no_grad():
        out=djgnet(c,bp)
    out=out.cpu()
    out=np.asarray(out)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa


lazy = 'vessel'
GROUND_TRUTH = scipy.io.loadmat(root_gt + lazy + '_true_100_100.mat')['BV2']
GROUND_TRUTH = cv2.resize(GROUND_TRUTH,(128,128),interpolation=cv2.INTER_NEAREST)
GROUND_TRUTH = zscorenorm(GROUND_TRUTH)
# GROUND_TRUTH = GROUND_TRUTH.T
num11 = GROUND_TRUTH.flatten()

c = scipy.io.loadmat(root_raw + lazy + '_sensor_data_100_100.mat')['sensor_data']
# c = cv2.resize(c,(128,512),interpolation=cv2.INTER_NEAREST)
# bp
aaa = scipy.io.loadmat(root_bp + lazy + '_sensor_data_100_100_bp.mat')['back2']
# aaa = cv2.resize(aaa,(128,128),interpolation=cv2.INTER_NEAREST)

pre = testpre(c,aaa)
num12 = pre.flatten()

p1 = pearsonr(num11,num12)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num12),2)
print(p1,p2)

fig, ax = plt.subplots(nrows=1, figsize=(6,6))
im = ax.imshow(pre, extent=[0, 1, 0, 1])
ax.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
cb=plt.colorbar(im)
plt.show()