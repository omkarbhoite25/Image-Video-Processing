# # # #########################
# # # Name: Omkar Bhoite
# # # Neptun ID: JJ349G
# # # #########################


# The code is being referred from  https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html , with some changes to tune it a little bit.




# # # KLT makes use of spatial intensity information to direct the search for the position that yields the best match.

# # # #######################################################################
# # #                     Pseudo Code#
# # # ########################################################################

# # # 1: Input: Two input images im1 and im2, pyramid level L
# # # 2: Output: Optical flow field f.
# # # 3: Generate Gaussian pyramids for im1 and im2
# # # 4: Initialize flow field f with zero values
# # # 5: for i = L -> 2 do
# # # 6: Compute the optical flow fi on pyramid level i using
# # # iterative Lucas-Kanade method with an initial guess=f
# # # 7: 2X bilinear interpolate fi
# # # in both height and width and
# # # store the result in f
# # # 8: end for
# # # 9: Compute the optical flow f1 on pyramid level 1 using
# # # iterative Lucas-Kanade method with an initial guess=f
# # # 10: f = f1

# # # ########################################################################


# # # The code below perform the KLT tracking and has been sourced from opencv implementation.


import numpy as np
import cv2 as cv
import argparse
a = input(" (1.Motion tracking or 2.Optical Dense Flow) Enter 1 or 2: ")
if (a == 1):
    parser = argparse.ArgumentParser(description='Assignment 4')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()
    cap = cv.VideoCapture(args.image)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter('output_video_motion_tracking.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1000,
                        qualityLevel = 0.2,
                        minDistance = 10,
                        blockSize = 1 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 0,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY) # convert rbg image fram captured from the video to gray scale.
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params) # determining the strong corners in an image
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame) # zeros array of same shape as the frame
    while(1):
        ret,frame = cap.read() # returns the status and  the fram captured from the video
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert rbg image fram captured from the video to gray scale.
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #calculating the optical flow of the image space using the KLT method and the returns the nextpoints from the image as well as the status and the error
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks based on the points tracked the old and the new one
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2) # mask to see the tracking operation taking place
            frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1) # display the points in form of circle
        img = cv.add(frame,mask) # add the frame and mask to get the complete tracking
        out.write(img) # to write the video and save it into the directory
        cv.imshow('frame',img) # display the video
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    out.release()






###############################################
#Dense Optical Flow
###############################################

# The algorithm analyzes the content of two images, previous and current, and writes an estimate of the motion to an output image.

# As shown below, the algorithm splits input images into  pixel blocks. 
# Then, for each block, it estimates the content translation from the previous to the current frame, 
# and writes the estimate as a motion vector to the corresponding pixel in the output image.
# The code below perform the dense optical flow and has been sourced from opencv implementation.


if (a ==2):
    parser = argparse.ArgumentParser(description='Assignment 4')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()
    # parser to parse the inputed video for further processing
    cap = cv.VideoCapture(args.image)
    frame_width = int(cap.get(3)) #frame width
    frame_height = int(cap.get(4))#frame height
    out = cv.VideoWriter('output_video_dense_optical_flow.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height)) # to save the video 
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY) # convert rbg image fram captured from the video to gray scale.
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 15, 15, 3, 5, 10, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        out.write(bgr)
        cv.imshow('frame2',bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png',frame2)
            cv.imwrite('opticalhsv.png',bgr)
        prvs = next
    out.release()