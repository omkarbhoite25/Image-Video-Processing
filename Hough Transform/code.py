import cv2
import numpy as np
from collections import defaultdict

def hough_circle_transform(image, edge_image, r_min, r_max, delta_r, theta_step_size, Threshold,pixel_Threshold):
  rows, cols = edge_image.shape[:2]
  accu = np.zeros((rows,cols))
  delta_theta = int(360 / theta_step_size)
  theta = np.arange(0, 360, step=delta_theta)
  R = np.arange(r_min, r_max, step=delta_r)
  cos_theta = np.cos(np.deg2rad(theta))
  sin_theta = np.sin(np.deg2rad(theta))
  circle_candidates = []
  for r in R:
    for t in range(theta_step_size):
      circle_candidates.append((r, int(r * cos_theta[t]), int(r * sin_theta[t])))
  accumulator = defaultdict(int)
  for y in range(rows):
    for x in range(cols):
      if edge_image[y][x] != 0: 
        for r, rcos_t, rsin_t in circle_candidates:
          x_center = x - rcos_t
          y_center = y - rsin_t
          accumulator[(x_center, y_center, r)] += 1
          # if x_center >= 0 and x_center < rows and y_center >= 0 and y_center < cols: 
          #   accu[x_center][y_center] = 1
  hough_circles = image.copy() 
  hough_circles_prime = image.copy()
  acc_cell_max = np.amax(accu)
  acc_cell_min = np.amin(accu)
  circles = []
  for circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = circle
    current_vote = votes / theta_step_size
    if current_vote >= Threshold:
      circles.append((x, y, r, current_vote))
      if x >= 0 and x < cols and y >= 0 and y < rows:
        accu[y][x] = 255
      # else:
      #   accu[y][x] = 0

  pixel_threshold = pixel_Threshold
  circ = []
  for x, y, r, v in circles:
    if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in circ):
      circ.append((x, y, r, v))
  circles = circ
  for x, y, r, v in circles:
    hough_circles_with_center = cv2.circle(hough_circles, (x,y), r, (0,0,255), 2)
    hough_circles_with_center = cv2.circle(hough_circles,(x,y),2,(0,255,0),3)
    hough_circles_without_center = cv2.circle(hough_circles_prime, (x,y), r, (0,0,255), 2)

  return hough_circles_with_center,accu, hough_circles_without_center

def main():
  img = input('Image: ')
  r_min = int(input('R minimum: '))  #By default use 5
  r_max = int(input('R maximum: '))  #By default use 30
  ksize = int(input('Ksize: '))   #By default use 5
  delta_r = int(input('Radius step size: '))  #By default use 1
  theta_step_size = int(input('Theta step size : '))  #By default use 100
  Threshold =float(input('Thershold: '))   #By default use 0.5
  PThreshold =float(input('Pixel Thershold: '))   #By default use 3
  input_img = cv2.imread(img)
  print ("Generating edge image")
  if ksize == 0:
    input_img = input_img
  else:
    input_img = cv2.bilateralFilter(input_img,ksize,50,50)
  edge_image = cv2.Canny(input_img,100, 200)
  # print(edge_image.shape)
  cv2.imshow('Edge Image', edge_image)
  print('Press any key to proceed')
  cv2.waitKey(0)
  cv2.imwrite("edge_image.png", edge_image)
  print ("Generating hough circle transform image")
  circle_img,accu, circle_img_without_center = hough_circle_transform(input_img, edge_image, r_min, r_max, delta_r, theta_step_size, Threshold,PThreshold)
  cv2.imshow('Hough Circle Image', circle_img)
  print('Press any key to proceed')
  cv2.waitKey(0)
  cv2.imwrite("hough.png", circle_img)

  print ("Generating hough circle image without center")
  cv2.imshow('Hough circle without center Image', circle_img_without_center)
  print('Press any key to proceed')
  cv2.waitKey(0)
  cv2.imwrite("hough_without_center.png", circle_img_without_center)

  print ("Generating accumulator image")
  cv2.imshow('Accumulator Image', accu)
  print('Press any key to proceed')
  cv2.waitKey(0)
  cv2.imwrite("accu_image.png", accu)
  print ("Hough Circle Transform Complete")



if __name__ == "__main__":
    main()
