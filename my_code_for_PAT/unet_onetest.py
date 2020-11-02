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
    # diff = img1 / 255. - img2#当第一个true是double，需要除255，预测的保存的是single，对single*255没有用.好像这样也不对
    diff = diff.flatten('C')#这样对图片也适用
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
import time

root_model = 'F:/unet_paper_e2e/pre28_6layers/72658/'
djgnet=torch.load(root_model + 'unet_paper.pkl',map_location='cpu')# djgnet=djgnet.module
djgnet=djgnet.to('cuda')

cax1 = 3
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

root_true = 'F:/train/train_dataset/new_machine/test/review/response/'
ground_truth = scipy.io.loadmat(root_true + 'pat_true_100_100_pyzscore.mat')['pat_true_100_100_pyzscore']
# ground_truth = scipy.io.loadmat(root_newBV2 + 'BV2.mat')['BV2']
# BV2  140*140 120*120 进行裁剪
# line1 = range(20)
# line2 = range(120,140)
# ground_truth = np.delete(ground_truth,line2,axis=1)
# ground_truth = np.delete(ground_truth,line2,axis=0)
# ground_truth = np.delete(ground_truth,line1,axis=0)
# ground_truth = np.delete(ground_truth,line1,axis=1)
# print(line1,line2,ground_truth.shape)
# BV2  80*80 60*60 进行填充
# padding_size = 10
# ground_truth = np.pad(ground_truth,((padding_size,padding_size),(padding_size,padding_size)),'constant',constant_values = (0,0))
ground_truth = cv2.resize(ground_truth,(128,128),interpolation=cv2.INTER_NEAREST)
# ground_truth = zscorenorm(ground_truth)
# ground_truth = np.rot90(ground_truth,3)#正数代表逆时针
# ground_truth = np.fliplr(ground_truth)
num11 = ground_truth.flatten()

savename = 'matlab'
root_testdata = 'F:/train/train_dataset/new_machine/test/review/response/'
c = scipy.io.loadmat(root_testdata + savename + '.mat')['sensor_data_40db']#sensor_data_40db
c = c.T
c = cv2.resize(c,(128,512),interpolation=cv2.INTER_NEAREST)
aaa = testpre(c)

root_save = root_testdata

scipy.io.savemat(root_save + savename + '_resunet.mat', {savename + '_resunet': aaa})
# c = scipy.io.loadmat('F:/train/train_dataset/new_machine/test/bp_collection/pat_BV2_100_100_fbp.mat')['reconstruction']
# aaa = cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)#0.99 16.23
# aaa = zscorenorm(aaa)
num12 = aaa.flatten()

p1 = pearsonr(num11,num12)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num12),2)
print(p1,p2)


fig, ax = plt.subplots(nrows=1, figsize=(6,6))
im = ax.imshow(aaa, extent=[0, 1, 0, 1],vmin=0, vmax=4)#,vmin=0, vmax=cax1
ax.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
position=fig.add_axes([0.91, 0.12, 0.03, 0.75])#left, bottom, width, height :bar距离左边，距离底部，bar的宽度，bar的高度
cb=plt.colorbar(im,cax=position)
plt.savefig(root_save + savename + '_resunet.png')
plt.show()


# fig, ((ax1),(ax2)) = plt.subplots(nrows=1, ncols=2,figsize=(25,25))#,figsize=(10,10)
# axlist = [(ax1),(ax2)]
# plt.suptitle('right: ground truth  left: pre', fontsize=25)
# # ax1.set_title()
# im1 = ax1.imshow(ground_truth,vmin=0, vmax=cax1)
# im2 = ax2.imshow(aaa,vmin=0, vmax=cax1)
# ax2.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
# position = fig.add_axes([0.92, 0.12, 0.01, 0.75])#left, bottom, width, height , pic=2*4
# plt.colorbar(mappable=im1,ax=axlist,cax=position)
# fig = plt.get_current_fig_manager().window.state('zoomed')
# # plt.savefig(root_save + savename + '_60_2.png')
# plt.show()