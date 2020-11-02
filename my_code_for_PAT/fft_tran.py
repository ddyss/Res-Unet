import numpy as np
from PIL import Image
import scipy.io
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
roottest = 'F:/train/train_dataset/new_machine/test/'
img = scipy.io.loadmat(roottest + 'disc_corner_sensor_data_100_100.mat')['sensor_data']
import scipy.signal as signal
img = img.T
def stft_pic(img):  #257*3直接拉伸成一列，立体图显示都是在一侧边缘处集中显示高频信息
    for i in range(128):
        begin = np.zeros(shape=(771,128))
        aaa = signal.stft(img[:,i],nperseg=512,nfft=512)
        begin[:,i] = aaa[2].reshape(-1)
        begin = cv2.resize(begin,(128,512),interpolation=cv2.INTER_NEAREST)
    return begin
def stft_pic1(img):  #257*3并排排在一起，立体图显示都是在40处集中显示高频信息
    for i in range(128):
        begin = np.zeros(shape=(257,128*3))
        aaa = signal.stft(img[:,i],nperseg=512,nfft=512)
        begin[:,i:i+3] = aaa[2]
        begin = cv2.resize(begin,(128,512),interpolation=cv2.INTER_NEAREST)
    return begin
aaa = stft_pic1(img)
plt.imshow(aaa)
plt.colorbar()
plt.show()
from mpl_toolkits.mplot3d import axes3d, Axes3D
size=aaa.shape
Y=np.arange(0,size[0],1)
X=np.arange(0,size[1],1)
X,Y=np.meshgrid(X,Y)
fig=plt.figure()
# ax=fig.gca(projection='3d')#0.99版本
ax = Axes3D(fig)#1.版本
ax.plot_surface(X,Y,aaa,cmap='rainbow')
plt.show()

'''
#快速傅里叶变换算法得到频率分布
f = np.fft.fft2(img)#2 means two dimensions
#默认结果中心点位置是在左上角,调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

#取绝对值：将复数变化成实数，绝对值结果是振幅
#取对数的目的为了将数据变化到较小的范围（比如0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift))
# ph_f = np.angle(f)
# ph_fshift = np.angle(fshift)
plt.subplot(121),plt.imshow(s1),plt.title('original'),plt.colorbar()
plt.subplot(122),plt.imshow(s2),plt.title('center'),plt.colorbar()
plt.show()
# '''

'''
#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

#将频谱低频从左上角移动至中心位置
dft_shift = np.fft.fftshift(dft)

#频谱图像双通道复数转换为0-255区间
result = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

#显示图像
plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(result, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
# '''