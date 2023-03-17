from pathlib import Path
import cv2 as cv
from enum import Enum
import pandas as pd
import os
import argparse
import sys

class State(Enum):
    VISIBLE = 0
    OCCLUDED = 1
    MOTION = 2
    NON_PLAY = 3

keybindings = {
    'next':          [ ord('l'), 3 ], # 3 = right arrow
    'prev':          [ ord('h'), 2 ], # 2 = left arrow
    'visible':       [ ord('v'), ],
    'occluded':      [ ord('o'), ],
    'motion':        [ ord('m'), ],
    'fast-forward':  [ ord('f'), ],
    'next_selection':[ ord('n'), ],
    'remove':        [ ord('x'), ],
    'circle_grow':   [ ord('='), ord('+') ],
    'circle_shrink': [ ord('-'), ],
    'quit':          [ ord('q'), ],
}

class VideoPlayer():
    def __init__(self, opt) -> None:
        self.cap = cv.VideoCapture(opt.video_path)
        self.width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.video_path = Path(opt.video_path)
        self.circle_size = 10
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
            self.info = {'frame_num': [], 'x': [], 'y': [], 'visible': []}
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
            if self.frame_num in self.info['frame_num']:
                num = self.info['frame_num'].index(self.frame_num)
                self.info['x'][num] = x
                self.info['y'][num] = y
                self.info['visible'][num] = self.state.value
            else:
                self.info['frame_num'].append(self.frame_num)
                self.info['x'].append(x)
                self.info['y'].append(y)
                self.info['visible'].append(self.state.value)
            self.clicked = True


    def display(self):
        res_frame = self.frame.copy()
        res_frame = cv.putText(res_frame, self.state.name, (100, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 2, self.colors[self.state.value], 2, cv.LINE_AA)
        if self.frame_num in self.info['frame_num']:
            num = self.info['frame_num'].index(self.frame_num)
            x = int(self.info['x'][num] * self.width)
            y = int(self.info['y'][num] * self.height)
            visible = self.info['visible'][num]
            cv.circle(res_frame, (x, y), self.circle_size, self.colors[visible], self.circle_size)
        cv.imshow('Frame', res_frame)
        self.clicked = False


    def run(self):
        key = cv.waitKeyEx(1)
        if key in keybindings['fast-forward']:
            if self.state == State.NON_PLAY:
                self.state = State.VISIBLE
            else:
                self.state = State.NON_PLAY
        if self.state == State.NON_PLAY:
            for _ in range(4):
                self.cap.grab()
            self.frame_num += 4
            self.clicked = True
        if key in keybindings['remove']:
            if self.frame_num in self.info['frame_num']:
                row_num = self.info['frame_num'].index(self.frame_num)
                self.info['x'].pop(row_num)
                self.info['y'].pop(row_num)
                self.info['frame_num'].pop(row_num)
                self.info['visible'].pop(row_num)
        if key in keybindings['next_selection']:
            info_df = pd.DataFrame.from_dict(self.info).sort_values(by=['frame_num'], ignore_index=True)
            tmp = info_df[info_df['frame_num'] > self.frame_num]
            if not tmp.empty:
                new_frame_num = tmp.iloc[0]['frame_num']
                self.cap.set(cv.CAP_PROP_POS_FRAMES, new_frame_num)
                self.frame_num = new_frame_num - 1
                self.clicked = True
            self.info = {k: list(v.values()) for k, v in info_df.to_dict().items()}
        if key in keybindings['circle_grow']:
            self.circle_size += 1
        if key in keybindings['circle_shrink']:
            self.circle_size -= 1
        if key in keybindings['occluded']:
            self.state = State.OCCLUDED
        if key in keybindings['motion']:
            self.state = State.MOTION
        if key in keybindings['visible']:
            self.state = State.VISIBLE
        if key in keybindings['next']:
            self.clicked = True
        if key in keybindings['prev']:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_num - 1)
            self.frame_num = int(self.cap.get(cv.CAP_PROP_POS_FRAMES) - 1)
            self.clicked = True
        if key in keybindings['quit']:
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
        df = pd.DataFrame.from_dict(self.info).sort_values(by=['frame_num'], ignore_index=True)
        df.to_csv(self.csv_path, index=False)


    def __del__(self):
        self.finish()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='?', default=None, help='Path to the video file.')
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
        if previous_frame is None or cv.PSNR(frame, previous_frame) < 40.:
            out.write(frame)

        # Update the previous frame
        previous_frame = frame
    print('finished removing duplicates')



if __name__ == '__main__':
    opt = parse_opt()

    # run as an executable
    if opt.video_path is None:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        p  = Path(application_path)
        video_path = next(p.glob('*.mp4'))
        toRemove = input('Should duplicated, consecutive frames be deleted? Insert "y" or "n": \n')
        if toRemove == 'y':
            bez_duplikatow_video_path = str(video_path.with_stem(video_path.stem + '_no_dups'))
            remove_duplicate_frames(str(video_path), bez_duplikatow_video_path)
            video_path = bez_duplikatow_video_path
        opt.video_path = str(video_path)

    # run as a CLI script
    elif opt.remove_duplicate_frames == True:
        remove_duplicate_frames(opt.video_path, opt.video_path)

    player = VideoPlayer(opt)
    while(player.cap.isOpened()):
        player.run()
