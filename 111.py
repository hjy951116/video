import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt

# # Open a vidoe
# video_path=r"./vtest.mp4"
 

# img_path =r'./images'
# img_path2 =r'./equimages'

# if not os.path.isdir(img_path):
#    mkdir(img_path)

# # Divide the video into frames
# vidcap = cv2.VideoCapture(video_path)
# (cap,frame)= vidcap.read()
 
# if cap==False:
#     print('cannot open video file')
# count = 0

# # Save the frames in a folder
# while cap:
#   cv2.imwrite(os.path.join(img_path,'%.6d.jpg'%count),frame)

#   count += 1
#   # Every 100 frames get 1
#   for i in range(1):
#     (cap,frame)= vidcap.read()
count2 = 0
framey = []
deltay = []
deltab = []
deltab2 = []
i=0
a = 1
# previous_gray = cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY)
previous_y = numpy.mean(cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY))
previous_blocky = numpy.zeros(1947)
previous_blocky2 = numpy.zeros(1947)
# Open frames in the folder
import glob
from PIL import Image
for frames in glob.glob('./images/*.jpg'):
  img = cv2.imread(frames)
  gray_levels = 256
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   a = gray/previous_gray
#   previous_gray = gray 
  equ = cv2.equalizeHist(gray)
  # cv2.imwrite(os.path.join(img_path2,'%.6d.jpg'%count2),equ)
  # count2 += 1
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]

  # Define the window size
  windowsize_r = 32
  windowsize_c = 32
#   m = int(height/16)
#   n = int(width/16)
  # The average luminance component (Y) of an entire frame
  # for m in range(width):
  #     for n in range(height):
  #       r, g, b = pix[m, n]
  #       y = (0.2126*r + 0.7152*g + 0.0722*b)
  #       #print(m,n)
  #       #print(y)
  y1 = numpy.mean(gray)
  framey.append(y1)
  biiiii = numpy.power(y1-previous_y,2)
  biiiiii = numpy.sum(biiiii)
  deltay.append(biiiiii)
  previous_y = y1
  #print(y1)
  print(i)
  # print(framey)
  # print(deltay)
  
  blocky = []
  blocky2 = []
 
  # Each frame is partitioned into blocks
  # for r in range(0,gray.shape[0] - windowsize_r, windowsize_r):
  #   for c in range(0,gray.shape[1] - windowsize_c, windowsize_c):
  #       window = gray[r:r+windowsize_r,c:c+windowsize_c]
  #       hist = numpy.histogram(window,bins=gray_levels)
  #       # The average luminance component of each block 
  #       w = numpy.mean(window)
  #       blocky.append (w)
 
  #       # The blocks are sorted in decreasing order 
  #       w1 = numpy.sort(blocky)
  #       #print(window)
  #       #print(blocky)
  
  # b = numpy.array(w1)-previous_blocky
  # bs = numpy.power(b,2)
  # deltab.append(sum(numpy.power(b,2)))
  # # print(previous_blocky)
  # previous_blocky = numpy.array(w1)
  # # print(previous_blocky)
  # print(deltab)
  # print(b)
  # plt.plot(w1) # plotting by columns
  # plt.show()
  #print(blocky,w1)
  
  # Each frame is partitioned into blocks
  for r2 in range(0,equ.shape[0] - windowsize_r, windowsize_r):
    for c2 in range(0,equ.shape[1] - windowsize_c, windowsize_c):
        window2 = equ[r2:r2+windowsize_r,c2:c2+windowsize_c]
        hist2 = numpy.histogram(window2,bins=gray_levels)
        # The average luminance component of each block 
        w2 = numpy.mean(window2)
        blocky2.append(w2)
        # The blocks are sorted in decreasing order 
        w3 = numpy.sort(blocky2)
        #print(blocky2)
  b2 = numpy.array(w3)-previous_blocky2
  deltab2.append(sum(numpy.power(b2,2)))
  # print(previous_blocky2)
  previous_blocky2 = numpy.array(w3)
  # print(previous_blocky2)
  # print(deltab2)
  # plt.plot(b2) # plotting by columns
  # plt.show()
  i = i+1
                #print(hist)
# print(framey)
# plt.plot(framey) # plotting by columns
# # plt.xticks(range(len(framey)))
# plt.show()
# print(y2)
# print(deltay)
# plt.plot(deltay) # plotting by columns
# plt.show()
