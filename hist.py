import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('000071.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])
 
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
 
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
cv2.imwrite('img2.jpg',img2)
histm,binsm = np.histogram(img2.flatten(),256,[0,256])
 
cdfm = histm.cumsum()
cdfm_normalized = cdfm * histm.max()/ cdfm.max()
 
plt.plot(cdfm_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdfm','histogram'), loc = 'upper left')
plt.show()
