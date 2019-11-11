import cv2
import numpy as np
import os

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
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)
        return ret, bgr


if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "Crew_1280x720_60Hz.yuv"
    size = (720, 1280)
    cap = VideoCaptureYUV(filename, size)
    count = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            # cv2.imwrite('%.1d.png'%count,frame)
            cv2.imwrite(os.path.join('./test','%.1d.png'%count),frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(30)
            count += 1
        else:
            break
