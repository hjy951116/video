import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys
import pandas as pd

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 // 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420, 3)
        return ret, bgr

with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))


if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "Crew_1280x720_60Hz.yuv"
    size = (720, 1280)
    cap = VideoCaptureYUV(filename, size)
    ret, frame = cap.read()
    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    count = 0
    # Open frames in the folder
    for n in range(10):
        previous_img = frame.copy()
        prev_gray = cv.cvtColor(previous_img, cv.COLOR_BGR2GRAY)
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Opens a new window and displays the input frame
        # cv.imshow("input", frame)
        # cv.imshow("prev", previous_img)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
        RV=np.arange(720,3,-4)
        CV=np.arange(3,1280,4)
        u,v = np.meshgrid(CV, RV)
        print(n)
        fig, ax = plt.subplots()
        x = flow[..., 0][::4, ::4]
        y = flow[..., 1][::4, ::4]
        q = ax.quiver(u,v,x, y,color='red',headlength=5)
        plt.show()
