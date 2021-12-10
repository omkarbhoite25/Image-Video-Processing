import numpy as np 
import cv2
from numpy.core.fromnumeric import shape
from numpy.lib.histograms import histogram
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")



class otsu():
    def rgb2gray(rgb):
        red, green, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue

        return gray

input_image = cv2.imread('phobos.png')
Gray = otsu.rgb2gray(input_image)
BINS = np.array(range(0,257))
P_i, pixels =np.histogram(input_image, BINS)
P_i = P_i.astype(float)
for i in range(len(P_i)):
    if P_i[i] == 0:
        P_i[i] = 1e-10
totalCount = np.sum(P_i)
mean = np.sum((P_i[1:])*pixels[1:256])/np.sum(P_i[1:])
sigma_beta = []
for i in range(1,256):
    q_1 = np.sum(P_i[:i])/totalCount
    q_2 = np.sum(P_i[i:])/totalCount
    mean_c1 = (np.sum(P_i[:i]*pixels[:i]))/np.sum(P_i[:i])
    variance_c1 = np.sum(((pixels[:i]-mean_c1)**2)*P_i[:i])/np.sum(P_i[:i])
    mean_c2 = np.sum(P_i[i:]*pixels[i:256])/np.sum(P_i[i:])
    variance_c2 = np.sum(((pixels[i:256]-mean_c2)**2)*P_i[i:])/np.sum(P_i[i:])
    sigma_beta.append(q_1*q_2*((mean_c1-mean_c2)**2))

t_optimal = np.argmax(sigma_beta)
print(t_optimal)
plt.plot(sigma_beta)
plt.axvline(t_optimal, color='red',linestyle='--',linewidth=1)
plt.show()

for i in range(3):
    for j in range(input_image.shape[0]):
        for k in range(input_image.shape[1]):
            if input_image[j,k,i] < t_optimal:
                input_image[j,k,i] = 0
            else:
                input_image[j,k,i] = 255

cv2.imwrite('otsu.jpg', input_image)
otsu = cv2.imread('otsu.jpg')
cv2.imshow('otsu ',otsu)


