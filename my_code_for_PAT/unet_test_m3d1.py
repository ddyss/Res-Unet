import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
import scipy
from scipy.stats import pearsonr
import math
# psnr = skimage.measure.compare_psnr(im1, im2, 255)
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
def re_zscorenorm(truth,now):
    bb = np.mean(truth)
    cc = truth.std()
    aaa = now * cc + bb
    return aaa
def max_min(x):
    aa = np.max(x)
    bb = np.min(x)
    new = (x-bb)/(aa-bb)
    return new
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
    c=c.T
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

root_test = 'C:/Users/27896/Desktop/paper/conven_testdata/'
# root_test = 'F:/train/train_dataset/new_machine/test/'
lazy2020 = 'pat'
SENSORDATA=scipy.io.loadmat(root_test + lazy2020 + '_sensor_data.mat')['sensor_data_40db']

root_true = 'F:/train/train_dataset/new_machine/test/box/recon0724/'
GROUND_TRUTH=scipy.io.loadmat(root_true + lazy2020 + '_true_100_100.mat')['BV2']
GROUND_TRUTH=cv2.resize(GROUND_TRUTH,(128,128),interpolation=cv2.INTER_NEAREST)
GROUND_TRUTH = zscorenorm(GROUND_TRUTH)
# GROUND_TRUTH = max_min01(GROUND_TRUTH)


# root_save = 'C:/Users/27896/Desktop/paper/conven_testdata/model_pre_dere/'


cax1 = 4# 九宫球 = 8  光声字母缩写 = 4 血管 = 10  分辨率 = 2.5
showcre = True
model1 = 'F:/unet_paper_e2e/pre34r2unet/'
savename1 = 'r2_unet'
savetitle1 = 'R2_Unet'
model2 = 'F:/unet_paper_e2e/pre30asym_ori/final_best/'
savename2 = 'sta_unet'
savetitle2 = 'Sta_Unet'
model3 = 'F:/unet_paper_e2e/pre21nested/final_best/'             #'F:/unet_paper_e2e/pre29cnn/U_cnn/final_best/'
savename3 = 'unet_plus'
savetitle3 = 'Unet++'
model4 = 'F:/unet_paper_e2e/pre26asym_inception/final_best_real/'
savename4 = 'incep_unet'
savetitle4 = 'Incep_Unet'
model5 = 'F:/unet_paper_e2e/pre22denseskip/dense4_debug/final_best_real/'
savename5 = 'ds_unet'
savetitle5 = 'Ds_Unet'
model6 = 'F:/unet_paper_e2e/pre31denseblock/debug6_derive/final_best_real/'
savename6 = 'db_unet'
savetitle6 = 'Db_Unet'
model7 = 'F:/unet_paper_e2e/pre28_6layers/72658/'
savename7 = 'res_unet'
savetitle7 = 'Res_Unet(proposed)'

fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4,figsize=(25,25))#,figsize=(10,10)
axlist = [(ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)]
# plt.suptitle('up: predict  down: true', fontsize=25)


num11 = GROUND_TRUTH.flatten()
# scipy.io.savemat(root_save + 'ground_truth.mat', {'ground_truth': GROUND_TRUTH})
ax1.set_title('Ground truth')
im1 = ax1.imshow(GROUND_TRUTH,vmin=0, vmax=cax1)
# """
djgnet=torch.load(model1 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename1 + '.mat', {savename1: aaa})
num12 = aaa.flatten()
ax2.set_title(savetitle1)
p1 = pearsonr(num11,num12)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num12),2)
if showcre:
    ax2.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im2 = ax2.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model2 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename2 + '.mat', {savename2: aaa})
num13 = aaa.flatten()
ax3.set_title(savetitle2)
p1 = pearsonr(num11,num13)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num13),2)
if showcre:
    ax3.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im3 = ax3.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model3 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre_nested(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename3 + '.mat', {savename3: aaa})
num14 = aaa.flatten()
ax4.set_title(savetitle3)
p1 = pearsonr(num11,num14)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num14),2)
if showcre:
    ax4.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im4 = ax4.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model4 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename4 + '.mat', {savename4: aaa})
num15 = aaa.flatten()
ax5.set_title(savetitle4)
p1 = pearsonr(num11,num15)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num15),2)
if showcre:
    ax5.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im5 = ax5.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model5 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename5 + '.mat', {savename5: aaa})
num16 = aaa.flatten()
ax6.set_title(savetitle5)
p1 = pearsonr(num11,num16)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num16),2)
if showcre:
    ax6.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im6 = ax6.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model6 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename6 + '.mat', {savename6: aaa})
num17 = aaa.flatten()
ax7.set_title(savetitle6)
p1 = pearsonr(num11,num17)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num17),2)
if showcre:
    ax7.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im7 = ax7.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model7 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
# aaa = max_min01(aaa)
# scipy.io.savemat(root_save + savename7 + '.mat', {savename7: aaa})
num18 = aaa.flatten()
ax8.set_title(savetitle7)
p1 = pearsonr(num11,num18)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num18),2)
if showcre:
    ax8.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im8 = ax8.imshow(aaa,vmin=0, vmax=cax1)

# ax8.set_title('')
# plt.axis('off')
# """
# position = fig.add_axes([0.92, 0.35, 0.01, 0.3])#left, bottom, width, height , pic=1*5
position = fig.add_axes([0.92, 0.12, 0.01, 0.75])#left, bottom, width, height , pic=2*4
plt.colorbar(mappable=im1,ax=axlist,cax=position)
fig = plt.get_current_fig_manager().window.state('zoomed')
plt.show()
# '''
