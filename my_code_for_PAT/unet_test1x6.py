import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
import scipy
from scipy.stats import pearsonr
import math
# psnr = skimage.measure.compare_psnr(im1, im2, 255)
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
    print(c.shape)
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    print(out.shape)
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
    print(c.shape)
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    print(out.shape)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa
root_testdata100 = 'F:/train/train_dataset/new_machine/test/'
root_savedata100 = 'D:/profile/temp/'
root_bprecon = 'F:/train/train_dataset/new_machine/test/bp_collection/'
lazy1 = 'dere'
GROUND_TRUTH=scipy.io.loadmat(root_testdata100 + lazy1 + '_true_100_100.mat')['BV2']
GROUND_TRUTH=cv2.resize(GROUND_TRUTH,(128,128),interpolation=cv2.INTER_NEAREST)
SENSORDATA=scipy.io.loadmat(root_testdata100 + lazy1 + '_sensor_data_100_100.mat')['sensor_data']
BP_RECON=scipy.io.loadmat(root_bprecon + lazy1 + '_sensor_data_100_100_bp.mat')['back2']#TR_cropto100
BP_RECON = cv2.resize(BP_RECON,(128,128),interpolation=cv2.INTER_NEAREST)
cax1 = 100# 九宫球 = 8  光声字母缩写 = 4 血管 = 10  分辨率 = 2.5
showcre = True
# model1 = 'F:/unet_paper_e2e/pre29cnn/U_cnn/final_best/'
# savename1 = 'cnn'
# savetitle1 = 'CNN'
model2 = 'F:/unet_paper_e2e/pre30asym_ori/final_best/'
savename2 = 'standard_unet'
savetitle2 = 'Standard_Unet'
model3 = 'F:/unet_paper_e2e/pre21nested/final_best/'
savename3 = 'unet_plus'
savetitle3 = 'Unet++'
# model4 = 'F:/unet_paper_e2e/pre26asym_inception/final_best_real/'
# savename4 = 'inception_unet'
# savetitle4 = 'Inception_Unet'
# model5 = 'F:/unet_paper_e2e/pre22denseskip/dense4_debug/final_best_real/'
# savename5 = 'dense_skip_unet'
# savetitle5 = 'Dense_skip_Unet'
model6 = 'F:/unet_paper_e2e/pre31denseblock/debug6_derive/final_best_real/'
savename6 = 'dense_block_unet'
savetitle6 = 'Dense_block_Unet'
model7 = 'F:/unet_paper_e2e/pre28_6layers/72658/'
savename7 = 'res_unet'
savetitle7 = 'Res_Unet(proposed)'

fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=1, ncols=6)#,figsize=(10,10)
axlist = [(ax1,ax2,ax3,ax4,ax5,ax6)]
# plt.suptitle('up: predict  down: true', fontsize=25)

aaa = GROUND_TRUTH
# aaa = zscorenorm(aaa)
# aaa = aaa * 0.1# * 0.1 变大， * 10 变小
num11 = aaa.flatten()
# scipy.io.savemat(root_savedata100 + 'ground_truth.mat', {'ground_truth': aaa})
ax1.set_title('Ground truth')
im1 = ax1.imshow(aaa,vmin=0, vmax=cax1)

aaa = BP_RECON
# aaa = zscorenorm(BP_RECON)
# scipy.io.savemat(root_savedata100 + savename1 + '.mat', {savename1: aaa})
num12 = aaa.flatten()
ax2.set_title('BP')
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
aaa = re_zscorenorm(GROUND_TRUTH,aaa)
# scipy.io.savemat(root_savedata100 + savename2 + '.mat', {savename2: aaa})
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
aaa = re_zscorenorm(GROUND_TRUTH,aaa)
# scipy.io.savemat(root_savedata100 + savename3 + '.mat', {savename3: aaa})
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

# djgnet=torch.load(model4 + 'unet_paper.pkl',map_location='cpu')
# djgnet=djgnet.to('cuda')
# aaa = testpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename4 + '.mat', {savename4: aaa})
# num15 = aaa.flatten()
# ax5.set_title(savetitle4)
# p1 = pearsonr(num11,num15)[0]
# p1 = round(p1,2)
# p2 = round(psnr(num11,num15),2)
# if showcre:
#     ax5.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
# else:
#     pass
# im5 = ax5.imshow(aaa,vmin=0, vmax=cax1)

# djgnet=torch.load(model5 + 'unet_paper.pkl',map_location='cpu')
# djgnet=djgnet.to('cuda')
# aaa = testpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename5 + '.mat', {savename5: aaa})
# num16 = aaa.flatten()
# ax6.set_title(savetitle5)
# p1 = pearsonr(num11,num16)[0]
# p1 = round(p1,2)
# p2 = round(psnr(num11,num16),2)
# if showcre:
#     ax6.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
# else:
#     pass
# im6 = ax6.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model6 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
aaa = re_zscorenorm(GROUND_TRUTH,aaa)
# scipy.io.savemat(root_savedata100 + savename6 + '.mat', {savename6: aaa})
num17 = aaa.flatten()
ax5.set_title(savetitle6)
p1 = pearsonr(num11,num17)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num17),2)
if showcre:
    ax5.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im5 = ax5.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model7 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = testpre(SENSORDATA)
aaa = re_zscorenorm(GROUND_TRUTH,aaa)
# scipy.io.savemat(root_savedata100 + savename7 + '.mat', {savename7: aaa})
num18 = aaa.flatten()
ax6.set_title(savetitle7)
p1 = pearsonr(num11,num18)[0]
p1 = round(p1,2)
p2 = round(psnr(num11,num18),2)
if showcre:
    ax6.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
else:
    pass
im6 = ax6.imshow(aaa,vmin=0, vmax=cax1)

# ax8.set_title('')
# plt.axis('off')#类似的操作想起作用，需要放在show之前，imshow之后

position = fig.add_axes([0.92, 0.39, 0.01, 0.21])#left, bottom, width, height , pic=1*6
# position = fig.add_axes([0.92, 0.12, 0.01, 0.75])#left, bottom, width, height , pic=2*4
plt.colorbar(mappable=im1,ax=axlist,cax=position)
fig = plt.get_current_fig_manager().window.state('zoomed')
plt.show()
# '''
