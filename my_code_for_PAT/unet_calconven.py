import matplotlib.pyplot as plt
import scipy.io
import torch
import math
import numpy as np
import cv2
from scipy.stats import pearsonr
def max_min01(aaa):
    amax = np.max(aaa)
    amin = np.min(aaa)
    aaa = (aaa - amin)/(amax - amin)
    return aaa
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
def psnr(img1, img2):
    # print(type(img1))#<class 'numpy.ndarray'>
    # print(type(img2))#<class 'numpy.ndarray'>
    diff = img1 - img2#-------------当后面max=255
    # diff = img1 / 255. - img2 / 255.#-------------当后面max=1；和上式计算结果一样
    diff = diff.flatten('C')#这样对图片也适用，且不会影响结果
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.uint8)#变不变float64都一样，变成uint8，值会略大，
    mse = np.mean(diff ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#计算结果和MATLAB一样
lazy = 'pat'
root_true = 'F:/train/train_dataset/new_machine/test/'
ground_truth = scipy.io.loadmat(root_true + lazy + '_true_100_100.mat')['BV2']
# ground_truth = cv2.resize(ground_truth,(128,128),interpolation=cv2.INTER_NEAREST)
ground_truth = zscorenorm(ground_truth)
# ground_truth = max_min01(ground_truth)
num11 = ground_truth.flatten()
print(np.max(ground_truth))
mrrreconname = 'recon_pat_sensor_data_20db_pyzscore'
# root_conven = 'F:/train/train_dataset/new_machine/test/box/recon0724/'
root_conven = 'C:/Users/27896/Desktop/paper/conven_testdata/fourtype_20_60recon/'
conven = scipy.io.loadmat(root_conven + mrrreconname + '.mat')['pat_20db']#sensor_data_40db
# root_conven = 'F:/train/train_dataset/new_machine/test/bp_collection/'
# conven = scipy.io.loadmat(root_conven + lazy + '_sensor_data_100_100_bp.mat')['back2']#sensor_data_40db
# conven = cv2.resize(conven,(128,128),interpolation=cv2.INTER_NEAREST)
# conven = max_min01(conven)
# conven = zscorenorm(conven)
num12 = conven.flatten()
print(np.max(conven))
p1 = pearsonr(num11,num12)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num12),2)
print(p1,p2)


# fig, ax = plt.subplots(nrows=1, figsize=(6,6))
# im = ax.imshow(conven, extent=[0, 1, 0, 1],vmin=0, vmax=3.5)#,vmin=0, vmax=cax1
# ax.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
# position=fig.add_axes([0.92, 0.12, 0.03, 0.3])#left, bottom, width, height :bar距离左边，距离底部，bar的宽度，bar的高度
# cb=plt.colorbar(im,cax=position)
# # plt.savefig(root_save + savename + '_60.png')
# plt.show()