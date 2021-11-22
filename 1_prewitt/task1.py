
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt




#######
#Load your image
#######
a = input('Input the image name along with the extension: ')
input_image = cv2.imread(a)
# print(input_image)
#######
#Convert RGB image into Gray 
#######
def rgb2gray(rgb):

    red, green, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return gray

gray = rgb2gray(input_image)
# # print(len(input_image))
cv2.imshow('Original Image',input_image)
# print(np.shape(gray))

########
#Perwitt Kernel matrix
########
perwitt_x = np.array([[-1 ,0, 1],[-1,0,1],[-1,0,1]])
perwitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])


########
#Convolution of Iput Image and Kernel to get the filtered image
########
def convolution(Image,Kernel):
    Image_height = Image.shape[0]
    Image_width = Image.shape[1]

    Kernel_height = Kernel.shape[0]
    Kernel_width = Kernel.shape[1]

    Output_matrix = np.zeros((Image_height,Image_width))

    Height = (Kernel_height-1 )//2
    Width = (Kernel_width-1) //2

    for i in np.arange(Height,Image_height-Height):
        for j in np.arange(Width,Image_width-Width):
            sum = 0
            for k in np.arange(-Height,Height+1):
                for l in np.arange(-Width,Width+1):
                    matrix_a = Image[i+k,j+l]
                    matrix_w = Kernel[Height+k,Width+l]
                    sum += (matrix_a*matrix_w)
            Output_matrix[i,j] = sum
    return Output_matrix

a = convolution(gray,perwitt_x)/3.0
b = convolution(gray,perwitt_y)/3.0
# print(a)


########
#Calculate the magnitude using the convolued images along the X & Y direction
########
perwitt_gardient_magnitude = np.sqrt(np.power(a,2)+np.power(b,2))
out = (perwitt_gardient_magnitude/ np.max(perwitt_gardient_magnitude))*255

# print(out)
cv2.imwrite('Gradient_Magnitude.jpg', out)
GM = cv2.imread('Gradient_Magnitude.jpg')
cv2.imshow('Gradient Magnitude',GM)


perwitt_angle = np.rad2deg(np.arctan2(b,a))+180
# print(perwitt_angle)
########
#Function to perform the Non-Maxima Suppression
########

def non_maximum_suppression(Input_Image, Perwitt_Angle):
    size = Input_Image.shape
    suppressed_matrix = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= Perwitt_Angle[i, j] < 22.5) or (337.5 <= Perwitt_Angle[i, j] <= 360) or (157.5 <= Perwitt_Angle[i, j] <= 202.5):
                compare_value = max(Input_Image[i, j - 1], Input_Image[i, j + 1])
            elif (22.5 <= Perwitt_Angle[i, j] < 67.5) or (202.5 <= Perwitt_Angle[i, j] < 247.5):
                compare_value = max(Input_Image[i - 1, j - 1], Input_Image[i + 1, j + 1])
            elif (67.5 <= Perwitt_Angle[i, j] < 112.5) or (247.5 <= Perwitt_Angle[i, j] < 292.5):
                compare_value = max(Input_Image[i - 1, j], Input_Image[i + 1, j])
            else:
                compare_value = max(Input_Image[i + 1, j - 1], Input_Image[i - 1, j + 1])
            if Input_Image[i, j] >= compare_value:
                suppressed_matrix[i, j] = Input_Image[i, j]
    suppressed_matrix = np.multiply(suppressed_matrix, 255.0 / suppressed_matrix.max())
    return suppressed_matrix

NMS_image = non_maximum_suppression(out,perwitt_angle)
cv2.imwrite('NMS.jpg', NMS_image)
NMS = cv2.imread('NMS.jpg')
cv2.imshow('NMS',NMS)


#######
#Please increase the wait key if you want to view the result for longertime.
#######
cv2.waitKey(20000)
cv2.destroyAllWindows()
