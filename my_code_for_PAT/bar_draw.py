import matplotlib.pyplot as plt
import numpy as np
x1 = [20,40,60]
x1 = np.array(x1)
#pc
# y1 = [0.63,0.55,0.78,0.71,0.81,0.57,0.83,0.8]
# y2 = [0.66,0.68,0.89,0.89,0.93,0.84,0.90,0.95]
# y3 = [0.67,0.68,0.89,0.88,0.93,0.83,0.88,0.94]
#psnr
y1 = [1.27,1.56,3.41,2.90,4.26,1.67,3.97,4.46]
y2 = [1.73,2.67,5.01,5.65,7.92,5.33,6.72,10.06]
y3 = [1.74,2.67,5.12,5.32,7.49,5.15,6.03,9.26]
comhe = [y1,y2,y3]
comhe = np.array(comhe)
comhe = comhe.T
r1 = comhe[0][:]
r2 = comhe[1][:]
r3 = comhe[2][:]
r4 = comhe[3][:]
r5 = comhe[4][:]
r6 = comhe[5][:]
r7 = comhe[6][:]
r8 = comhe[7][:]
width = 2
ax = plt.subplot(1,1,1)
ax.bar(x1, r1, width, color='c',label = 'BP')
ax.bar(x1+width, r2, width, color='g',label = 'Sta_Unet')
ax.bar(x1+width * 2, r3, width, color='k',label = 'Incep_Unet')
ax.bar(x1+width * 3, r4, width, color='y',label = 'Ds_Unet')
ax.bar(x1+width * 4, r5, width, color='m',label = 'Db_Unet')
ax.bar(x1+width * 5, r6, width, color='b',label = 'Unet++')
ax.bar(x1+width * 6, r7, width, color='0.8',label = 'R2_Unet')
ax.bar(x1+width * 7, r8, width, color='r',label = 'Res_Unet')
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(x1+7)#7刚好
ax.set_xticklabels(['20dB', '40dB', '60dB'],fontsize = 15)
ax.set_yticklabels(['0', '2', '4', '6', '8', '10'],fontsize = 15)
plt.xlabel('(b)',fontsize = 25)
plt.ylabel('PSNR',fontsize = 20)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.11),ncol=8,fontsize = 12)
plt.get_current_fig_manager().window.state('zoomed')
plt.show()
