import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import math
import csv

with open('./test2.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

with open('./test3.csv','rb') as csvfile1:
  reader1 = csv.reader(csvfile1)
  column1 = [row[1] for row in reader1]
  column1.pop(0)
  column1 = list(map(int,column1))

# Open a vidoe
video_path=r"./vtest.mp4"
 

img_path =r'./images'
img_path2 =r'./equimages'

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
deltaya = []
deltayb = []
deltayc = []
deltab = []
deltab2 = []
i = 0
j = 0
a = 1
frameindex = []
xa = []
xb = []
xc = []
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
  equ = cv2.equalizeHist(gray)
  cv2.imwrite(os.path.join(img_path2,'%.6d.jpg'%count2),equ)
  count2 += 1
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
#  biii = y1-previous_y
#   if biii >= 3:
#       print(j)
#   j += 1
  bmax = max(y1, previous_y)
  bi = numpy.power(y1-previous_y,2)
  bii = numpy.sum(bi)
  biii = math.sqrt(bii)
  deltay.append(bii)
  #print(y1,previous_y)  
  #P = abs(y1-previous_y)/bmax
  previous_y = y1

  #print(i)
  #print(P)
  
  # if bii >= 100:
  # #   i = i+1
  #   print(j)
  # #print(y1)
  # #print(i)
  # #print(y1)
  # j = j+1
  
  
  blocky = []
  blocky2 = []
 
  # # Each frame is partitioned into blocks
  # for r in range(0,gray.shape[0] - windowsize_r, windowsize_r):
  #   for c in range(0,gray.shape[1] - windowsize_c, windowsize_c):
  #       window = gray[r:r+windowsize_r,c:c+windowsize_c]
  #       hist = numpy.histogram(window,bins=gray_levels)
  #       # The average luminance component of each block 
  #       w = numpy.mean(window)
  #       blocky.append (w)
 
  #       # The blocks are sorted in decreasing order 
  #       w1 = numpy.sort(blocky)
  
  # b = numpy.array(w1)-previous_blocky
  # bs = numpy.power(b,2)
  # deltab.append(sum(numpy.power(b,2)))
  # previous_blocky = numpy.array(w1)
  
  # # Each frame is partitioned into blocks
  # for r2 in range(0,equ.shape[0] - windowsize_r, windowsize_r):
  #   for c2 in range(0,equ.shape[1] - windowsize_c, windowsize_c):
  #       window2 = equ[r2:r2+windowsize_r,c2:c2+windowsize_c]
  #       hist2 = numpy.histogram(window2,bins=gray_levels)
  #       # The average luminance component of each block 
  #       w2 = numpy.mean(window2)
  #       blocky2.append(w2)
  #       # The blocks are sorted in decreasing order 
  #       w3 = numpy.sort(blocky2)
  # b2 = numpy.array(w3)-previous_blocky2
  # deltab2.append(sum(numpy.power(b2,2)))
  # previous_blocky2 = numpy.array(w3)
  if column[i] == 1:
    xa.append(i)
    deltaya.append(deltay[i])
  elif column1[i] == 1:    
    xb.append(i)
    deltayb.append(deltay[i])
  else:
    xc.append(i)
    deltayc.append(deltay[i])
  i = i+1
n = len(deltay)
x = range(0,n)
for j in range (1,n-2):
  # if deltay[j]-deltay[j-1]>30 and deltay[j+1]deltay[j]>  and deltay[j+1]>deltay[j+2]:
  Hn0 = abs(deltay[j]- deltay[j-1])
  Hn = abs(deltay[j+1]- deltay[j])
  Hn1 = abs(deltay[j+2]- deltay[j+1])
  Hmax = max(Hn, Hn1)
  #print(Hn)
  if Hn < 20 and Hn0 > 30 and Hn1 > 30 and deltay[j]>deltay[j-1]:
  #print(Hn,Hn1,Hmax)
  # P = abs(Hn - Hn1)/Hmax
  # frameindex.append(1)
    print(j)
  #else:
   # frameindex.append(0)
#print(frameindex)
# plt.plot(deltab) # plotting by columns
# plt.plot(deltab2) # plotting by columns
# plt.plot(deltay) # plotting by columns
# plt.show()
# print(framey)
# arr = numpy.array(deltay)
# arr1 = numpy.array(deltab)
# arr2 = numpy.array(deltab2)
# getter = [68,71,89,126,134,143,147,151,160,161,162,163,170,171,190,191,199,200,213,269,270,271,331,339,340,348,397,398,409,434,465,466,467,545,558,559,561,562,563,564,565,566,567,568,570,574,577,578,579,585,586,604,612,640,644,646,647,648,649,650,651,653,661,662,663,664,665,666,667,668,674,675,711,712,713,714,715,716,721,722,723,724,725,727,728,729,730,738,739,740,741,744,745,761,762,766,769,778,779,780,782,783,784,785,786,787,795,798,799,800,802,803,804,813,814,815,817,818,819,823,825,842,843,844,848,849,852,853,870,871,872,874,875,876,884,885,886,888,889,890,912,913,916,917,963,964,965,966,968,969,970,972,973,976,977,980,981,982]
# print(arr[getter])
# print(arr1[getter])
# print(arr2[getter])
#plt.plot(deltay) # plotting by columns
# # plt.xticks(range(len(framey)))
#plt.show()
# print(y2)
# print(deltay)
plt.axhline(y = 35, color = 'g', linestyle = '-')
plt.axhline(y = 300, color = 'r', linestyle = '-')
plt.scatter(xa,deltaya,c='red')
#plt.scatter(xb,deltayb,c='green' )
plt.scatter(xc,deltayc)
plt.show()
