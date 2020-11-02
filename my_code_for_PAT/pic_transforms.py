from torchvision import transforms
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
data = scipy.io.loadmat('D:/test128/six2_sensor_data_128_128.mat')['sensor_data']
target = scipy.io.loadmat('D:/test128/six2_true_128_128.mat')['BV2']
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
# data = zscorenorm(data)

transform=transforms.Compose([
    transforms.RandomVerticalFlip(p=1),
    # transforms.ToTensor(),
])
transform_target=transforms.Compose([
    transforms.RandomRotation(degrees=(90,90)),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.ToTensor(),
])

newdata = transform(Image.fromarray(data))
newtarget = transform_target(Image.fromarray(target))
# newtarget = transform_target(target)
plt.imshow(data)
plt.colorbar()
plt.show()
plt.imshow(newdata)
plt.colorbar()
plt.show()
