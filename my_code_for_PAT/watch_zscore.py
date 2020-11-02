import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import scipy
from scipy.stats import pearsonr
import math
from nested import djgnet
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
root128 = 'D:/test128/diff_mag2/'
def max_min(x):
    aa = np.max(x)
    bb = np.min(x)
    new = (x-bb)/(aa-bb)
    return new
c=scipy.io.loadmat(root128 + 'mag20_true_128_128.mat')['BV2']
c = zscorenorm(c)
img = Image.fromarray(c)
plt.imshow(img)
plt.colorbar()
plt.show()