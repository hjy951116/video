import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys


with open('./test.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

count2 = 0
framey = []
i = 0
j = 0
k = 0
a = 0.2

P = []


# Open frames in the folder
preim = Image.open('000000.jpg')
prepix = preim.load()
for frames in glob.glob('./test/*.jpg'):
  img = cv2.imread(frames)
  gray_levels = 256
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  hist = numpy.histogram(gray,bins=gray_levels)
  
  # print(hist)
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]
  

  Ym = numpy.mean(gray)
  YA = 0.5 + (Ym - cv2.bilateralFilter(gray, d=5, sigmaColor=5, sigmaSpace=5))**3


  if column[k] == 1:
    for j in range (255):
      f = hist[0]
      P.append(float(f[j])/float(921600))
    Pl = sum()
    Ph = sum
  for x in range (720):
    for y in range (1280):
      if gray[x][y] < Ym:
        F[x][y] = YA[x][y]*(sum(pj(1:int(255-gray[x][y]))))/Pl
      else:
        F[x][y] = YA[x][y]+(1-YA[x][y])*(sum(pj(int(255-gray[x][y]):)))/Ph


     
    # plt.imshow(adjustedpixel, interpolation='nearest')
    # plt.show()
    im.show()
    prepix = pix


  k += 1


