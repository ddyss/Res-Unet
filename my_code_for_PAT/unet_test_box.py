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
def max_min01(aaa):
    amax = np.max(aaa)
    amin = np.min(aaa)
    aaa = (aaa - amin)/(amax - amin)
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
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
# root_model = 'F:/unet_paper_e2e/pre31denseblock/debug6_derive/final_best_real/'
# root_model = 'F:/unet_paper_e2e/pre34r2unet/'
# root_model = 'F:/unet_paper_e2e/pre26asym_inception/final_best_real/'
# root_model = 'F:/unet_paper_e2e/pre22denseskip/dense4_debug/final_best_real/'
# root_model = 'F:/unet_paper_e2e/pre21nested/final_best/'
# root_model = 'F:/unet_paper_e2e/pre30asym_ori/final_best/'
root_model = 'F:/unet_paper_e2e/pre28_6layers/72658/'
djgnet=torch.load(root_model + 'unet_paper.pkl',map_location='cpu')# djgnet=djgnet.module
djgnet=djgnet.to('cuda')
# roottest = 'F:/train/train_dataset/new_machine/test/box/'
roottest = 'C:/Users/27896/Desktop/paper/conven_testdata/'
rootsave = 'F:/train/train_dataset/new_machine/test/box/resubmit/'
lazy = 'res_unet'
# line = [i for i in range(129, 385)]

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
# roottest = 'C:/Users/27896/Desktop/paper/conven_testdata/fourtype_20_60recon/'
# ctest = scipy.io.loadmat(roottest + 'mrr_recon.mat')['mrr_recon']
ctest = scipy.io.loadmat(roottest + 'test300_sensor_data_40db.mat')['djg300data']
roottrue = 'C:/Users/27896/Desktop/paper/conven_testdata/'
ctrue = scipy.io.loadmat(roottrue + 'test300_true_pyzscore.mat')['test300_true_pyzscore']#新 true不是cell，换读法
for i in range(300):
    # c = scipy.io.loadmat(roottest + 'test300_70.mat')['djg']#原来c和c覆盖了，导致报错，而原先循环里重复读取，所以没事
    c = ctest[0][i]
    aaa = testpre(c)
    # scipy.io.savemat(rootsave + 't' + str(i) + '.mat',{'t' + str(i):aaa})

    # c = max_min01(aaa)
    num11 = aaa.flatten()

    c = ctrue[i][:][:]
    c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
    # c = max_min01(c)#true 已经归一化了，不用再次zscore
    num22 = c.flatten()

    p1 = pearsonr(num11, num22)[0]
    p1 = round(p1, 2)
    pc_total.append(p1)
    p2 = round(psnr(num11, num22), 2)
    psnr_total.append(p2)

scipy.io.savemat(rootsave + 'pc_' + lazy + '.mat',{'pc_' + lazy : pc_total})
scipy.io.savemat(rootsave + 'psnr_' + lazy + '.mat',{'psnr_' + lazy : psnr_total})
print(pc_total)
print(psnr_total)
# np.save(roottest + 'pc.npy', pc_total)
# np.save(roottest + 'psnr.npy', psnr_total)