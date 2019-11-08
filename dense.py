import cv2 as cv
import numpy as np
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt


with open('./test2.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))
k = 0
count = 0
windowsize_r = 4
windowsize_c = 4


# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("test.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
previous_img = frame
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(frame)
# Sets image saturation to maximum
mask[..., 1] = 255

x = np.arange(0,1280)
y = np.arange(0,720)

# while(cap.isOpened()):
for n in range(10):
    previous_img = frame.copy()
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    # cv.imshow("input", frame)
    # cv.imshow("prev", previous_img)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
    RV=np.arange(720,3,-4)
    CV=np.arange(3,1280,4)
    u,v = np.meshgrid(CV, RV)
    print(k)
    fig, ax = plt.subplots()
    x = flow[..., 0][::4, ::4]
    y = flow[..., 1][::4, ::4]
    q = ax.quiver(u,v,x, y,color='red',headlength=5)
    plt.show()
    mask1=q

    # mag, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # RV=np.arange(0,720,10)
    # CV=np.arange(0,1280,10)
    # # These give arrays of points to sample at increments of 5
    # if count==0:
    #     count =1 #so that the following creation is only done once
    #     [Y,X]=np.meshgrid(CV,RV)
    #     # makes an x and y array of the points specified at sample increments

    # temp =mag[np.ix_(RV,CV)]
    # # this makes a temp array that stores the magnitude of flow at each of the sample points

    # motionvectors=np.array((Y[:],X[:],Y[:]+temp.real[:],X[:]+temp.imag[:]))

    # Ydist=motionvectors[0,:,:]- motionvectors[2,:,:]
    # Xdist=motionvectors[1,:,:]- motionvectors[3,:,:]
    # Xoriginal=X-Xdist
    # Yoriginal=Y-Ydist



    # plot2 = plt.figure()
    # plt.quiver(Xoriginal, Yoriginal, X, Y,
    #            color='red',
    #            headlength=1)

    # plt.title('Quiver Plot, Single Colour')
    # plt.show(plot2)
    # mask1 = plot2

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # print(magnitude, angle)
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # img = cv.add(frame,mask)
    # cv.imshow('name',img)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    img = frame
    imgcomp = img
    # imgcomp = img
    a = []
    b = []
    mapx= np.random.random((720,1280))
    mapy= np.random.random((720,1280))
    if column[k+1] == 1:
      for m in range (720):
        for n in range (1280):
          flow[m, n, 1] = - flow[m, n, 1] + m
          flow[m, n, 0] = - flow[m, n, 0] + n
      imgcomp = cv.remap(previous_img, flow[..., 0], flow[..., 1], interpolation = cv.INTER_CUBIC)
      # bicubic_img = cv.resize(previous_img,None, fx = 4, fy = 4, interpolation = cv.INTER_CUBIC)
      # bicubic_flow = cv.resize(flow,None, fx = 4, fy = 4, interpolation = cv.INTER_CUBIC)
      # cv.imwrite('bicubic_img.jpg',bicubic_img)
      # imgcomp = bicubic_img 
      # for m in range (720*4):
      #   for n in range (1280*4):
      #     x_frac = math.modf(abs(bicubic_flow[m, n, 0]))[0]
      #     y_frac = math.modf(abs(bicubic_flow[m, n, 1]))[0]
      #     if 0 <= x_frac < 0.125:
      #       bicubic_flow[m, n, 0] = int(bicubic_flow[m, n, 0])
      #     elif 0.125 <= x_frac < 0.375:
      #       bicubic_flow[m, n, 0] = int(bicubic_flow[m, n, 0]) + 0.25
      #     elif 0.375 <= x_frac < 0.625:
      #       bicubic_flow[m, n, 0] = int(bicubic_flow[m, n, 0]) + 0.5
      #     elif 0.625 <= x_frac < 0.875:
      #       bicubic_flow[m, n, 0] = int(bicubic_flow[m, n, 0]) + 0.75
      #     elif 0.875 <= x_frac < 1:
      #       bicubic_flow[m, n, 0] = int(bicubic_flow[m, n, 0]) + 1
      #     if 0 <= y_frac < 0.125:
      #       bicubic_flow[m, n, 1] = int(bicubic_flow[m, n, 1])
      #     elif 0.125 <= y_frac < 0.375:
      #       bicubic_flow[m, n, 1] = int(bicubic_flow[m, n, 1]) + 0.25
      #     elif 0.375 <= y_frac < 0.625:
      #       bicubic_flow[m, n, 1] = int(bicubic_flow[m, n, 1]) + 0.5
      #     elif 0.625 <= y_frac < 0.875:
      #       bicubic_flow[m, n, 1] = int(bicubic_flow[m, n, 1]) + 0.75
      #     elif 0.875 <= y_frac < 1:
      #       bicubic_flow[m, n, 1] = int(bicubic_flow[m, n, 1]) + 1
      # #     for r in range(m * windowsize_r, m * windowsize_r + windowsize_r): # bicubic_img.shape[1], 
      # #       for c in range(n * windowsize_c, n * windowsize_c+ windowsize_c):
      # #         if 0<= r + int(4*flow[m, n, 1]) < 2880 and 0<= c + int(4*flow[m, n, 0]) < 5120:
      # #           imgcomp[r, c] = bicubic_img[r + int(4*flow[m, n, 1]), c + int(4*flow[m, n, 0])]
      #     if 0<= m + int(4*bicubic_flow[m, n, 1]) < 2880 and 0<= n + int(4*bicubic_flow[m, n, 0]) < 5120:
      #       imgcomp[m, n] = bicubic_img[m + int(4*bicubic_flow[m, n, 1]), n + int(4*bicubic_flow[m, n, 0])]
      #     # a.append(int(4*flow[m, n, 1]))
      #     # b.append(int(4*flow[m, n, 0]))
      #     a.append(bicubic_flow[m, n, 1])
      #     b.append(bicubic_flow[m, n, 0])
          # c.append(magnitude[m, n])
          # d.append(angle[m, n])
          # interpolationflow = bicubic(flow[m, n, 0], flow[m, n, 1], img)

          # print(interpolationflow)
          # if int(m-flow[m, n, 1]) < 720 and int(m-flow[m,n,1]) >= 0 and int(n-flow[m,n,0]) < 1280 and int(n-flow[m,n,0]) >= 0:
          #   imgcomp[m,n,2] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),2]
          #   imgcomp[m,n,1] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),1]
          #   imgcomp[m,n,0] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),0]

      print(k)

      raw_data = {'y': a, 
      'x': b}
      # ,
      # 'magnitude': c,
      # 'angle': d}
      df = pd.DataFrame(raw_data , columns = ['y', 'x']) #, 'magnitude', 'angle'])
      df.to_csv('example.csv', index=False)
      # imgcomp1 = pic = cv.resize(imgcomp, (1280, 720), interpolation=cv.INTER_CUBIC)
      cv.imwrite('%.1d-comp.jpg'%count,imgcomp)
    count += 1
    k += 1
      

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
