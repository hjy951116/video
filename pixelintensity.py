import cv2
import os
import pickle

h= open('record.txt', 'w')
video_path=r"./vtest.mp4"
 

img_path=r'./images'
 

if not os.path.isdir(img_path):
   mkdir(img_path)
 

vidcap = cv2.VideoCapture(video_path)
(cap,frame)= vidcap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
if cap==False:
    print('cannot open video file')
count = 0


while cap:
  cv2.imwrite(os.path.join(img_path,'%.6d.jpg'%count),frame)

  count += 1
  for i in range(100):
    (cap,frame)= vidcap.read()

import glob
from PIL import Image
for frames in glob.glob('./images/*.jpg'):
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]
  for x in range(width):
      for y in range(height):
          r, g, b = pix[x, y]
          intensity= pix[x,y]
          ycbcr = im.convert('YCbCr')
          y, cb, cr = ycbcr.split()
          y.show()
      #print(r,g,b)          
      pickle.dump(intensity,h)
h.close()
