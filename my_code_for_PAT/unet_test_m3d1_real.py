# coding = UTF-8 # 一行原来是中文注释的意思
import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
import scipy
from scipy.stats import pearsonr
import math
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
def realpre_nested(c):
    # c = c.T
    c = cv2.resize(c, (128, 2048), interpolation=cv2.INTER_NEAREST)
    line = []
    for i in range(513, 1537):
        line.append(i)
    c = c[line, :]#只选取出指定行
    c = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
    c = zscorenorm(c)
    c = c.astype(np.float32)
    c = torch.tensor(c)
    c = c.reshape(1, 1, 128, 128)
    c = c.to('cuda')
    with torch.no_grad():
        out = djgnet(c)
    out = out.cpu()
    out = np.asarray(out)
    aaa = out.reshape(128, 128)
    aaa = aaa.T
    return aaa
def realpre(c):
    # c = c.T #不用转置，真值本身就是2030*128的形式，转置后变得肯定更差啊
    c = cv2.resize(c, (128, 512), interpolation=cv2.INTER_NEAREST)
    c = zscorenorm(c)
    c = c.astype(np.float32)
    c = torch.tensor(c)
    c = c.reshape(1, 1, 512, 128)
    c = c.to('cuda')
    with torch.no_grad():
        out = djgnet(c)
    out = out.cpu()
    out = np.asarray(out)
    aaa = out.reshape(128, 128)
    aaa = aaa.T
    return aaa

pathraw = 'F:/Study_CuSO4/Scan_1/deal_t1/'
pathraw_bprecon = pathraw + 'RECONs/'
SENSORDATA=scipy.io.loadmat(pathraw + 'del_threshold_smooth.mat')['smoo']#2030 x 128
# start_up = SENSORDATA[0:399]
# end_400 = SENSORDATA[1630:2029]
# SENSORDATA[0:399] = SENSORDATA[1630:2029]#前400行用后400行替换
# BP_recon = scipy.io.loadmat(pathraw_bprecon+ 't1.mat')['t1']
# root_savedata100 = 'D:/profile_old/real/debug/'
cax1 = 5# 3 na     2 cuni     5 cuso4

model1 = 'F:/unet_paper_e2e/pre34r2unet/'
savename1 = 'r2_unet'
savetitle1 = 'R2_Unet'
model2 = 'F:/unet_paper_e2e/pre30asym_ori/final_best/'
savename2 = 'sta_unet'
savetitle2 = 'Sta_Unet'
model3 = 'F:/unet_paper_e2e/pre21nested/final_best/'
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

djgnet=torch.load(model1 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename1 + '.mat', {savename1: aaa})
ax1.set_title(savetitle1)
im1 = ax1.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model2 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename2 + '.mat', {savename2: aaa})
ax2.set_title(savetitle2)
im2 = ax2.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model3 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre_nested(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename3 + '.mat', {savename3: aaa})
ax3.set_title(savetitle3)
im3 = ax3.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model4 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename4 + '.mat', {savename4: aaa})
ax4.set_title(savetitle4)
im4 = ax4.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model5 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename5 + '.mat', {savename5: aaa})
ax5.set_title(savetitle5)
im5 = ax5.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model6 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename6 + '.mat', {savename6: aaa})
ax6.set_title(savetitle6)
im6 = ax6.imshow(aaa,vmin=0, vmax=cax1)

djgnet=torch.load(model7 + 'unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
aaa = realpre(SENSORDATA)
# scipy.io.savemat(root_savedata100 + savename7 + '.mat', {savename7: aaa})
ax7.set_title(savetitle7)
im7 = ax7.imshow(aaa,vmin=0, vmax=cax1)

# aaa = BP_recon.T
# aaa = cv2.resize(aaa, (128, 128), interpolation=cv2.INTER_NEAREST)
# aaa = zscorenorm(aaa)
# ax8.set_title('BP')
# im8 = ax8.imshow(aaa,vmin=0, vmax=cax1)

ax8.set_title('')
plt.axis('off')

# position = fig.add_axes([0.92, 0.35, 0.01, 0.3])#left, bottom, width, height , pic=1*5
position = fig.add_axes([0.92, 0.12, 0.01, 0.75])#left, bottom, width, height , pic=2*4
plt.colorbar(mappable=im1,ax=axlist,cax=position)
fig = plt.get_current_fig_manager().window.state('zoomed')
plt.show()

