import matplotlib.pyplot as plt
import scipy.io
import torch
import numpy as np
import time
def zscorenorm(aaa):
    bb = np.mean(aaa)
    cc = aaa.std()
    aaa = (aaa - bb)/cc
    return aaa
roottest = 'F:/train/train_dataset/new_machine/test/'
root_model = 'F:/unet_paper_e2e/pre28_6layers/72658/'
djgnet = torch.load(root_model + 'unet_paper.pkl',map_location='cpu')
djgnet = djgnet.cuda()
def testpre(c):
    c = c.T
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
lazy = 'nine1'
c = scipy.io.loadmat(roottest + lazy + '_sensor_data_100_100.mat')['sensor_data']
start = time.time()
aaa = testpre(c)
end = time.time()
print(end - start)
fig, ax = plt.subplots(nrows=1, figsize=(6,6))
im = ax.imshow(aaa, extent=[0, 1, 0, 1])
cb=plt.colorbar(im)
plt.show()
