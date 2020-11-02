import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
from PIL import Image
import scipy
from scipy.stats import pearsonr
import math
def psnr(img1, img2):
    diff = img1 - img2
    # diff = img1 / 255. - img2 / 255.#notnorm
    diff = diff.flatten('C')#这样对图片也适用
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = np.max(img2)#注意顺序不能颠倒，这里的后面是真值
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
def re_zscorenorm(truth,now):
    bb = np.mean(truth)
    cc = truth.std()
    aaa = now * cc + bb
    return aaa
root_model = 'F:/unet_paper_e2e/pre28_6layers/resunet_6layers_notop/'
djgnet=torch.load(root_model + 'unet_paper53.pkl',map_location='cpu')# djgnet=djgnet.module
djgnet=djgnet.to('cuda')
root_testdata = 'C:/Users/27896/Desktop/paper/conven_testdata/'#新的测试数据变了，真值可以不用变，归一化后能一样
root_truthdata = 'F:/train/train_dataset/new_machine/test/box/recon0724/'
root_save = root_model#'C:/Users/27896/Desktop/paper/conven_testdata/DL_recon/'
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
name1 = 'nine1'
name2 = 'pat'
name3 = 'vessel'
name4 = 'dere'
test_data1 = scipy.io.loadmat(root_testdata + name1 + '_sensor_data.mat')['sensor_data_40db']
test_data2 = scipy.io.loadmat(root_testdata + name2 + '_sensor_data.mat')['sensor_data_40db']
test_data3 = scipy.io.loadmat(root_testdata + name3 + '_sensor_data.mat')['sensor_data_40db']
test_data4 = scipy.io.loadmat(root_testdata + name4 + '_sensor_data.mat')['sensor_data_40db']

truth_data1 = scipy.io.loadmat(root_truthdata + name1 + '_true_100_100.mat')['BV2']
truth_data2 = scipy.io.loadmat(root_truthdata + name2 + '_true_100_100.mat')['BV2']
truth_data3 = scipy.io.loadmat(root_truthdata + name3 + '_true_100_100.mat')['BV2']
truth_data4 = scipy.io.loadmat(root_truthdata + name4 + '_true_100_100.mat')['BV2']
# '''
####figure(1)
plt.subplot(241)
c=test_data1
pre1 = testpre(c)
scipy.io.savemat(root_save + name1 + '_recon.mat', {name1 + '_recon.mat': pre1})
num11 = pre1.flatten()
img = Image.fromarray(pre1)
plt.imshow(img,vmin=0,vmax=8)
plt.colorbar()

plt.subplot(242)
c=test_data2
pre2 = testpre(c)
scipy.io.savemat(root_save + name2 + '_recon.mat', {name2 + '_recon.mat': pre2})
num12 = pre2.flatten()
img = Image.fromarray(pre2)
plt.imshow(img,vmin=0,vmax=4)
plt.colorbar()

plt.subplot(243)
c=test_data3
pre3 = testpre(c)
scipy.io.savemat(root_save + name3 + '_recon.mat', {name3 + '_recon.mat': pre3})
num13 = pre3.flatten()
img = Image.fromarray(pre3)
plt.imshow(img,vmin=0,vmax=10)
plt.colorbar()

plt.subplot(244)
c=test_data4
pre4 = testpre(c)
scipy.io.savemat(root_save + name4 + '_recon.mat', {name4 + '_recon.mat': pre4})
num14 = pre4.flatten()
img = Image.fromarray(pre4)
plt.imshow(img,vmin=0,vmax=2.5)
plt.colorbar()

plt.subplot(245)
c=truth_data1
c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
c = zscorenorm(c)
num21 = c.flatten()
img = Image.fromarray(c)
plt.imshow(img,vmin=0,vmax=8)
plt.colorbar()
p1 = pearsonr(num11,num21)[0]
p1 = round(p1,4)
p2 = round(psnr(num11,num21),2)
plt.xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')

plt.subplot(246)
c=truth_data2
c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
c = zscorenorm(c)
num22 = c.flatten()
img = Image.fromarray(c)
plt.imshow(img,vmin=0,vmax=4)
plt.colorbar()
p1 = pearsonr(num12,num22)[0]
p1 = round(p1,4)
p2 = round(psnr(num12,num22),2)
plt.xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')

plt.subplot(247)
c=truth_data3
c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
c = zscorenorm(c)
num23 = c.flatten()
img = Image.fromarray(c)
plt.imshow(img,vmin=0,vmax=10)
plt.colorbar()
p1 = pearsonr(num13,num23)[0]
p1 = round(p1,4)
p2 = round(psnr(num13,num23),2)
plt.xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')

plt.subplot(248)
c=truth_data4
c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
c = zscorenorm(c)
num24 = c.flatten()
img = Image.fromarray(c)
plt.imshow(img,vmin=0,vmax=2.5)
plt.colorbar()
p1 = pearsonr(num14,num24)[0]
p1 = round(p1,4)
p2 = round(psnr(num14,num24),2)
plt.xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
fig = plt.get_current_fig_manager().window.state('zoomed')
# plt.savefig('F:/unet_paper_e2e/pre19/5层/noblue/1.png')
plt.suptitle('up: predict  down: true', fontsize=14)
plt.show()
# '''





