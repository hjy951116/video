
import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys
import pandas as pd


with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))


deltay = numpy.zeros((720,1280))
deltau = numpy.zeros((720,1280))
deltav = numpy.zeros((720,1280))
ye = numpy.zeros((720,1280))
ue = numpy.zeros((720,1280))
ve = numpy.zeros((720,1280))


k = 0
cap = cv2.VideoCapture("test.mp4")
ret, frame = cap.read()

# image = cv2.imread('./test/000000.jpg')
# # gray_levels = 256
# # previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# yuv1 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# prev_gray = yuv1[...,0]
# prev_U = yuv1[...,1]  #0.492 * (img[...,0] - gray)
# prev_V = yuv1[...,2]
# # previous_U = 0.492 * (image[...,0] - previous_gray)
# # previous_V = 0.877 * (image[...,2] - previous_gray)


count = 0
fps = 5   
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  
videoWriter = cv2.VideoWriter('./testcomp.mp4', fourcc, fps, (1280,720))   
# Open frames in the folder
for n in range(10):
  prev_img = frame.copy()
  a = []
  b = []
  c = []
  ret, frame = cap.read()
  img = frame
#   gray_levels = 256
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  gray = yuv[...,0]
  U = yuv[...,1]  #0.492 * (img[...,0] - gray)
  V = yuv[...,2]  #0.877 * (img[...,2] - gray)

  prev_yuv = cv2.cvtColor(prev_img, cv2.COLOR_BGR2YUV)
  prev_gray = prev_yuv[...,0]
  prev_U = prev_yuv[...,1]
  prev_V = prev_yuv[...,2]
  for i in range (720):
    for j in range (1280):
    #   a.append(gray[i][j])
    #   b.append(previous_gray[i][j])
      if gray[i][j] > prev_gray[i][j]:
        deltay[i][j] = gray[i][j] - prev_gray[i][j]
        ye[i][j] = 1
      else:
        deltay[i][j] = prev_gray[i][j] - gray[i][j]
        ye[i][j] = -1
      if U[i][j] > prev_U[i][j]:
        deltau[i][j] = U[i][j] - prev_U[i][j]
        ue[i][j] = 1
      else:
        deltau[i][j] = prev_U[i][j] - U[i][j]
        ue[i][j] = -1
      if V[i][j] > prev_V[i][j]:
        deltav[i][j] = V[i][j] - prev_V[i][j]
        ve[i][j] = 1
      else:
        deltav[i][j] = prev_V[i][j] - V[i][j]
        ve[i][j] = -1
    #   c.append(ye[i][j]*deltay[i][j])
        


  # print(hist)
  # im = frame
  # pix = im.load()
  width = 1280
  height = 720
  pixelindex = numpy.zeros((720,1280))
  imgcomp = prev_img
  if column[k] == 1:
    # print(prev_img)
    for y in range (720):
      for x in range (1280):
        # pixelindex[m][n] = ye[m][n]*deltay[m][n]
        # pixelindex[m][n] = ue[m][n]*deltau[m][n]
        pixelindex[y][x] = ve[y][x]*deltav[y][x]
        if deltay[y][x] > 10 and ye[y][x] > 0 and ve[y][x] < 0 and deltau[y][x] > 4 and ue[y][x] > 0:          
          imgcomp[y,x,2] =  prev_img[y,x,2] - ye[y][x]*deltay[y][x] - ve[y][x]*1.14*deltav[y][x]
          imgcomp[y,x,1] =  prev_img[y,x,1] - ye[y][x]*deltay[y][x] + ue[y][x]*0.395*deltau[y][x] + ve[y][x]*0.581*deltav[y][x]
          imgcomp[y,x,0] =  prev_img[y,x,0] - ye[y][x]*deltay[y][x] - ue[y][x]*2.033*deltau[y][x]
        

    print(k,'flash')
    print(imgcomp)
    cv2.imwrite('%.1d-comp.jpg'%count,imgcomp)
    videoWriter.write(prev_img)
    plt.imshow(numpy.flipud(pixelindex),interpolation='nearest',cmap='bone',origin='lower')
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.show()
  else:
    print(k,'no flash')
#    cv2.imshow('img', img12)
#    cv2.waitKey(1000/int(fps))
    videoWriter.write(prev_img)

    

     
    # plt.imshow(adjustedpixel, interpolation='nearest')
    # plt.show()
    # im.show()


  count += 1
  k += 1
videoWriter.release()

    # while inputVideo.isOpened():
    #     ret, frame = inputVideo.read()

    #     if ret is True:
    #         # Calculate brightness of current frame
    #         frameBrightness = get_brightness(frame)

    #         # Maintain a buffer of the past 3 frames
    #         buffer.append(frameBrightness)
    #         if len(buffer) > BUFFER_SIZE:
    #             buffer.pop(0)

    #         # Detect if flash
    #         if detect_if_flash(buffer):
    #             print("Flash detected at frame %d" % count)

    #         count += 1

    #         outputVideo.write(frame)

    #     else:
    #         print("stream failed to read")
    #         break

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print("stream ended")
    #         break