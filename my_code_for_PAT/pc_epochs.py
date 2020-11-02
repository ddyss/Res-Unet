import scipy.io
import torch
from scipy.stats import pearsonr
import math
import numpy as np
import cv2
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
root_model = '/data1/MIP1/dataset/segnet/'
sensor_data = scipy.io.loadmat(root_model + 'vessel_sensor_data_100_100.mat')['sensor_data']
GROUND_TRUTH=scipy.io.loadmat(root_model + 'vessel_true_100_100.mat')['BV2']
GROUND_TRUTH=cv2.resize(GROUND_TRUTH,(128,128),interpolation=cv2.INTER_NEAREST)
num11 = GROUND_TRUTH.flatten()
def testpre(c):
    c=c.T
    c = zscorenorm(c)
    c=c.astype(np.float32)
    c=torch.tensor(c)
    c=c.reshape(1,1,512,128)  #数量  通道数
    c=c.to('cuda')
    # print(c.shape)
    with torch.no_grad():
        out=djgnet(c)
    out=out.cpu()
    out=np.asarray(out)
    # print(out.shape)
    aaa=out.reshape(128,128)
    aaa = aaa.T
    return aaa
pc_epochs = []
psnr_epochs = []
for epoch in range(33):
    djgnet = torch.load(root_model + 'unet_paper' + str(epoch) +'.pkl', map_location='cpu')
    djgnet = djgnet.cuda()
    aaa = testpre(sensor_data)
    num12 = aaa.flatten()
    p1 = pearsonr(num11, num12)[0]
    p1 = round(p1, 2)
    pc_epochs.append(p1)
    p2 = round(psnr(num11, num12), 2)
    psnr_epochs.append(p2)
np.save('pc_epochs.npy', pc_epochs)
np.save('psnr_epochs.npy', psnr_epochs)
# print(p1,p2)