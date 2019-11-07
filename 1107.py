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

image = cv2.imread('./test/000000.jpg')
# gray_levels = 256
# previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
yuv1 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
previous_gray = yuv1[...,0]
previous_U = yuv1[...,1]  #0.492 * (img[...,0] - gray)
previous_V = yuv1[...,2]
# previous_U = 0.492 * (image[...,0] - previous_gray)
# previous_V = 0.877 * (image[...,2] - previous_gray)


count = 0
fps = 5   
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  
videoWriter = cv2.VideoWriter('./testcomp.mp4', fourcc, fps, (1280,720))
# Open frames in the folder
for frames in glob.glob('./test/*.jpg'):
  a = []
  b = []
  c = []
  img = cv2.imread(frames)
#   gray_levels = 256
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  gray = yuv[...,0]
  U = yuv[...,1]  #0.492 * (img[...,0] - gray)
  V = yuv[...,2]  #0.877 * (img[...,2] - gray)

  for i in range (720):
    for j in range (1280):
    #   a.append(gray[i][j])
    #   b.append(previous_gray[i][j])
      if gray[i][j] > previous_gray[i][j]:
        deltay[i][j] = gray[i][j] - previous_gray[i][j]
        ye[i][j] = 1
      else:
        deltay[i][j] = previous_gray[i][j] - gray[i][j]
        ye[i][j] = -1
      if U[i][j] > previous_U[i][j]:
        deltau[i][j] = U[i][j] - previous_U[i][j]
        ue[i][j] = 1
      else:
        deltau[i][j] = previous_U[i][j] - U[i][j]
        ue[i][j] = -1
      if V[i][j] > previous_V[i][j]:
        deltav[i][j] = V[i][j] - previous_V[i][j]
        ve[i][j] = 1
      else:
        deltav[i][j] = previous_V[i][j] - V[i][j]
        ve[i][j] = -1
    #   c.append(ye[i][j]*deltay[i][j])
        
  previous_gray = gray
  previous_U = U
  previous_V = V

  # print(hist)
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]
  pixelindex = numpy.zeros((720,1280))

  if column[k] == 1:
    for m in range (720):
      for n in range (1280):
        # pixelindex[m][n] = ye[m][n]*deltay[m][n]
        # pixelindex[m][n] = ue[m][n]*deltau[m][n]
        pixelindex[m][n] = ve[m][n]*deltav[m][n]

        if deltay[m][n] > 10 and ye[m][n] > 0 and ve[m][n] < 0 and deltau[m][n] > 4 and ue[m][n] > 0:          
          img[m,n,2] =  img[m,n,2] - ye[m][n]*deltay[m][n] - ve[m][n]*1.14*deltav[m][n]
          img[m,n,1] =  img[m,n,1] - ye[m][n]*deltay[m][n] + ue[m][n]*0.395*deltau[m][n] + ve[m][n]*0.581*deltav[m][n]
          img[m,n,0] =  img[m,n,0] - ye[m][n]*deltay[m][n] - ue[m][n]*2.033*deltau[m][n]
        # elif deltay[m][n] > 40:
        #   img[m,n,2] =  img[m,n,2] - deltay[m][n] + 1.14*10
        #   img[m,n,1] =  img[m,n,1] - deltay[m][n] + 0.395*8 - 0.581*10
        #   img[m,n,0] =  img[m,n,0] - deltay[m][n] - 2.033*8
        # elif deltay[m][n] > 10:
        #   img[m,n,2] =  img[m,n,2] - deltay[m][n] + 1.14*4
        #   img[m,n,1] =  img[m,n,1] - deltay[m][n] + 0.395*8 - 0.581*4
        #   img[m,n,0] =  img[m,n,0] - deltay[m][n] - 2.033*8
    # dataframe = pd.DataFrame({'a_name':a,'b_name':b,'c_name':c})
    # dataframe.to_csv("result.csv",index=False,sep=',')
    print(k,'flash')
    cv2.imwrite('%.1d-comp.jpg'%count,img)
    videoWriter.write(img)
    plt.imshow(numpy.flipud(pixelindex),interpolation='nearest',cmap='bone',origin='lower')
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.show()
  else:
    print(k,'no flash')
    videoWriter.write(img)

    

     
    # plt.imshow(adjustedpixel, interpolation='nearest')
    # plt.show()
    # im.show()


  count += 1
  k += 1
videoWriter.release()