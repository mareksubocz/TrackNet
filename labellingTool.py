import cv2 as cv
import numpy as np
import sys
from enum import Enum
import pandas as pd
import os

class State(Enum):
    VISIBLE = 0
    OCCLUDED = 1
    MOTION = 2
    NON_PLAY = 3


class VideoPlayer():
    def __init__(self, video_path) -> None:
        self.cap = cv.VideoCapture(video_path)
        self.video_path = video_path
        self.window = cv.namedWindow('Frame', cv.WINDOW_AUTOSIZE)
        # cv.setWindowProperty('Frame', cv.WND_PROP_TOPMOST, 1)
        self.state = State.VISIBLE
        _, self.frame = self.cap.read()
        self.frame_num = 0
        self.clicked = False
        self.x = None
        self.y = None
        if os.path.exists(self.video_path + '.csv'):
            self.info = pd.read_csv(self.video_path+'.csv').to_dict()
            self.info = {k: list(v.values()) for k, v in self.info.items()}
            print(self.info)
        else:
            self.info = {'num': [], 'x': [], 'y': [], 'visible': []}
        self.colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,255)
        ]
        cv.setMouseCallback('Frame',self.markBall)
        self.display()


    def markBall(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.frame_num in self.info['num']:
                num = self.info['num'].index(self.frame_num)
                self.info['x'][num] = x
                self.info['y'][num] = y
                self.info['visible'][num] = self.state.value
            else:
                self.info['num'].append(self.frame_num)
                self.info['x'].append(x)
                self.info['y'].append(y)
                self.info['visible'].append(self.state.value)
            self.clicked = True


    def display(self):
        res_frame = self.frame.copy()
        res_frame = cv.putText(res_frame, self.state.name, (100, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 2, self.colors[self.state.value], 2, cv.LINE_AA)
        if self.frame_num in self.info['num']:
            num = self.info['num'].index(self.frame_num)
            x = self.info['x'][num]
            y = self.info['y'][num]
            visible = self.info['visible'][num]
            cv.circle(res_frame, (x, y), 2, self.colors[visible], 2)
        cv.imshow('Frame', res_frame)
        self.clicked = False


    def run(self):
        key = cv.waitKeyEx(1)
        if key == ord('n'):
            if self.state == State.NON_PLAY:
                self.state = State.VISIBLE
            else:
                self.state = State.NON_PLAY
        if self.state == State.NON_PLAY:
            for _ in range(4):
                self.cap.grab()
            self.frame_num += 4
            self.clicked = True
        if key == ord('o'):
            self.state = State.OCCLUDED
        if key == ord('m'):
            self.state = State.MOTION
        if key == ord('v'):
            self.state = State.VISIBLE
        if key == ord('l'):
            self.clicked = True
        if key == ord('h'):
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_num - 1)
            self.frame_num = int(self.cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
            self.clicked = True
        if key == ord('q'):
            self.finish()
            return
        if self.clicked:
            ret, self.frame = self.cap.read()
            self.x = None
            self.y = None
            self.frame_num += 1
            if not ret:
                self.finish()
                return
        self.display()


    def finish(self):
        self.cap.release()
        cv.destroyAllWindows()
        df = pd.DataFrame.from_dict(self.info)
        df.to_csv(self.video_path+'.csv', index=False)


    def __del__(self):
        self.finish()



if __name__ == '__main__':
    player = VideoPlayer(sys.argv[1])
    while(player.cap.isOpened()):
        player.run()
