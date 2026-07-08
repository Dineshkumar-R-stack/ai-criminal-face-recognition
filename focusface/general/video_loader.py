import os
from abc import *
from .configs import *
import cv2 as cv

class Loader(metaclass=ABCMeta):
    @abstractmethod
    def get_frame(self):
        pass

class StreamLoader(Loader):
    def __init__(self, vid_res, downsample):
        self._cap = cv.VideoCapture("http://192.168.42.43:8080/video")
        '''http://192.168.145.230:8080/video'''
        if not self._cap.isOpened():
            print("ERROR: Cannot open video stream")
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv.CAP_PROP_FPS, 30)  
        if vid_res == "adaptive":
            if downsample != 1:
                print("[Alert] opt[--vid-res] is \'adaptive\', so opt[down] is disabled.")
        else:
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, AVAILABLE_RESOLUTIONS[vid_res][1])
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, AVAILABLE_RESOLUTIONS[vid_res][0])
    def get_frame(self):
        return self._cap.read()

class VideoLoader(Loader):
    def __init__(self, vid_res, downsample, video_path, label_path):
        vid_list = os.listdir(video_path)
        label_list = os.listdir(label_path)
        trimmed_label_list = [label_name.replace('.txt', '') for label_name in label_list]
        if vid_res == "adaptive":
            if downsample != 1:
                print("[Alert] opt[--vid-res] is \'adaptive\', so opt[down] is disabled.")
        else:
            # 해상도를 세팅합니다.
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, AVAILABLE_RESOLUTIONS[vid_res][1])
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, AVAILABLE_RESOLUTIONS[vid_res][0])
    def get_frame(self):
        return self._cap.read()