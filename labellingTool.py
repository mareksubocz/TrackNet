from pathlib import Path
import cv2 as cv
import numpy as np
import sys
from enum import Enum
import pandas as pd
import os
import argparse
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

class State(Enum):
    VISIBLE = 0
    OCCLUDED = 1
    MOTION = 2
    NON_PLAY = 3


class VideoPlayer():
    def __init__(self, opt) -> None:
        self.cap = cv.VideoCapture(opt.video_path)
        self.width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
        self.video_path = Path(opt.video_path)
        if opt.csv_dir is None:
            self.csv_path = self.video_path.with_suffix('.csv')
        else:
            self.csv_path = Path(opt.csv_dir) / Path(self.video_path.stem).with_suffix('.csv')
        self.window = cv.namedWindow('Frame', cv.WINDOW_AUTOSIZE)
        # cv.setWindowProperty('Frame', cv.WND_PROP_TOPMOST, 1)
        self.state = State.VISIBLE
        _, self.frame = self.cap.read()
        self.frame_num = 0
        self.clicked = False
        self.x = None
        self.y = None
        if os.path.exists(self.csv_path):
            self.info = pd.read_csv(self.csv_path)
            if self.info['x'].max() > 1 or self.info['y'].max() > 1:
                self.info['x'] /= self.width
                self.info['y'] /= self.height
            self.info = {k: list(v.values()) for k, v in self.info.to_dict().items()}
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
        x /= self.width
        y /= self.height
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
            x = int(self.info['x'][num] * self.width)
            y = int(self.info['y'][num] * self.height)
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
        df.to_csv(self.csv_path, index=False)


    def __del__(self):
        self.finish()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--csv_dir', type=str, default=None, help='Path to the directory where csv file should be saved. If not specified, csv file will be saved in the same directory as the video file.')
    parser.add_argument('--remove_duplicate_frames', type=bool, default=False, help='Should identical consecutie frames be reduces to one frame.')
    opt = parser.parse_args()
    return opt


def remove_duplicate_frames(video_path, output_path):
    # Open the video file
    vid = cv.VideoCapture(video_path)

    # Set the frame width and height
    frame_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video file
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    # Read and process the frames one by one
    previous_frame = None
    while True:
        # Read the next frame
        success, frame = vid.read()

        # If we reached the end of the video, break the loop
        if not success:
            break

        # If the current frame is not a duplicate, write it to the output video
        if previous_frame is None or cv.PSNR(frame, previous_frame) < 32.:
            out.write(frame)

        # Update the previous frame
        previous_frame = frame
    print('finished removing duplicates')



if __name__ == '__main__':
    # remove_duplicate_frames('./dataset/videos/cut+2020.12.19-19.35-182088.mp4', './dataset/videos/NoDups1.mp4')
    opt = parse_opt()
    if opt.remove_duplicate_frames == True:
        remove_duplicate_frames(opt.video_path, opt.video_path)
    player = VideoPlayer(opt)
    while(player.cap.isOpened()):
        player.run()
