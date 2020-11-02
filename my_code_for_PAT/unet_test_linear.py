import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import cv2
import scipy
from scipy.stats import pearsonr
import math
def psnr(img1, img2):
    diff = img1 - img2
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
def max_min(x):
    aa = np.max(x)
    bb = np.min(x)
    new = (x-bb)/(aa-bb)
    return new
root128 = 'D:/test128/diff_mag2/'
# djgnet=torch.load('F:/unet_paper_e2e/pre21nested/final5/unet_paper.pkl',map_location='cpu')
djgnet=torch.load('F:/unet_paper_e2e/pre19new/6层/第二组/unet_paper.pkl',map_location='cpu')
djgnet=djgnet.to('cuda')
filters = [1, 2, 3, 4]
# line=[]
# for i in range(129,385):
#     line.append(i)
def testpre(c,num):
    c=c.T
    # c = c[line,:]
    # c=cv2.resize(c,(128,128),interpolation=cv2.INTER_NEAREST)
    c = zscorenorm(c)
    c = c*filters[num]
    c=c.astype(np.float32)
    c=torch.tensor(c)
    c=c.reshape(1,1,512,128)
    c=c.to('cuda')
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    print(out.shape)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa

#####figure(1)
fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4)
axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
plt.suptitle('up: predict    down: true', fontsize=25)
cax1 = 50

c=scipy.io.loadmat(root128 + 'mag1_sensor_data_128_128.mat')['sensor_data']
aaa = testpre(c,0)
# aaa = aaa*filters[0]
num11 = aaa.flatten()
#plt.clim(vmin=0, vmax=cax1)
im1 = ax1.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag2_sensor_data_128_128.mat')['sensor_data']
aaa = testpre(c,1)
# aaa = aaa*filters[1]
num12 = aaa.flatten()
#plt.clim(vmin=0, vmax=cax1)
im2 = ax2.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag3_sensor_data_128_128.mat')['sensor_data']
aaa = testpre(c,2)
# aaa = aaa*filters[2]
num13 = aaa.flatten()
#plt.clim(vmin=0, vmax=cax1)
im3 = ax3.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag4_sensor_data_128_128.mat')['sensor_data']
aaa = testpre(c,3)
# aaa = aaa*filters[3]
num14 = aaa.flatten()
#plt.clim(vmin=0, vmax=cax1)
im4 = ax4.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag1_true_128_128.mat')['BV2']
aaa = zscorenorm(c)
num21 = aaa.flatten()
aaa = aaa * filters[0]
p1 = pearsonr(num11,num21)[0]
p1 = round(p1,4)
p2 = round(psnr(num11,num21),2)
ax5.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
#plt.clim(vmin=0, vmax=cax1)
im5 = ax5.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag2_true_128_128.mat')['BV2']
aaa = zscorenorm(c)
num22 = aaa.flatten()
aaa = aaa * filters[1]
p1 = pearsonr(num12,num22)[0]
p1 = round(p1,4)
p2 = round(psnr(num12,num22),2)
ax6.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
#plt.clim(vmin=0, vmax=cax1)
im6 = ax6.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag3_true_128_128.mat')['BV2']
aaa = zscorenorm(c)
num23 = c.flatten()
aaa = aaa * filters[2]
p1 = pearsonr(num13,num23)[0]
p1 = round(p1,4)
p2 = round(psnr(num13,num23),2)
ax7.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
#plt.clim(vmin=0, vmax=cax1)
im7 = ax7.imshow(aaa,vmin=0, vmax=cax1)

c=scipy.io.loadmat(root128 + 'mag4_true_128_128.mat')['BV2']
aaa = zscorenorm(c)
num24 = c.flatten()
aaa = aaa * filters[3]
p1 = pearsonr(num14,num24)[0]
p1 = round(p1,4)
p2 = round(psnr(num14,num24),2)
ax8.set_xlabel(u'pearson:'+ str(p1) + u' psnr:'+ str(p2),fontproperties='SimHei')
#plt.clim(vmin=0, vmax=cax1)
im8 = ax8.imshow(aaa,vmin=0, vmax=cax1)

fig.colorbar(im1, ax=axlist)
plt.show()

