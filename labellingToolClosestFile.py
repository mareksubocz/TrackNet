from pathlib import Path
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, \
    WINDOW_AUTOSIZE, namedWindow, setMouseCallback, EVENT_LBUTTONDOWN, putText, \
    FONT_HERSHEY_SIMPLEX, imshow, destroyAllWindows, LINE_AA, circle, waitKeyEx, \
    CAP_PROP_POS_FRAMES, CAP_PROP_FRAME_COUNT, VideoWriter, VideoWriter_fourcc, \
    PSNR
from enum import Enum
from tqdm import tqdm
from pandas import read_csv, DataFrame
import os
import sys

class State(Enum):
    VISIBLE = 0
    OCCLUDED = 1
    MOTION = 2
    NON_PLAY = 3


class VideoPlayer():
    def __init__(self, video_path) -> None:
        self.cap = VideoCapture(video_path)
        self.width  = int(self.cap.get(CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(CAP_PROP_FRAME_HEIGHT)) 
        self.video_path = Path(video_path)
        self.csv_path = self.video_path.with_suffix('.csv')
        self.window = namedWindow('Frame', WINDOW_AUTOSIZE)
        # cv.setWindowProperty('Frame', cv.WND_PROP_TOPMOST, 1)
        self.state = State.VISIBLE
        _, self.frame = self.cap.read()
        self.frame_num = 0
        self.clicked = False
        self.x = None
        self.y = None
        if os.path.exists(self.csv_path):
            self.info = read_csv(self.csv_path)
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
        setMouseCallback('Frame',self.markBall)
        self.display()


    def markBall(self, event, x, y, flags, param):
        x /= self.width
        y /= self.height
        if event == EVENT_LBUTTONDOWN:
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
        res_frame = putText(res_frame, self.state.name, (100, 100),
                           FONT_HERSHEY_SIMPLEX, 2, self.colors[self.state.value], 2, LINE_AA)
        if self.frame_num in self.info['num']:
            num = self.info['num'].index(self.frame_num)
            x = int(self.info['x'][num] * self.width)
            y = int(self.info['y'][num] * self.height)
            visible = self.info['visible'][num]
            circle(res_frame, (x, y), 5, self.colors[visible], 2)
        imshow('Frame', res_frame)
        self.clicked = False


    def run(self):
        key = waitKeyEx(1)
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
            self.cap.set(CAP_PROP_POS_FRAMES, self.frame_num - 1)
            self.frame_num = int(self.cap.get(CAP_PROP_POS_FRAMES) - 1)
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
        destroyAllWindows()
        df = DataFrame.from_dict(self.info)
        df.to_csv(self.csv_path, index=False)


    def __del__(self):
        self.finish()


def remove_duplicate_frames(video_path, output_path):
    # Open the video file
    vid = VideoCapture(video_path)
    vidlength = int(vid.get(CAP_PROP_FRAME_COUNT))

    # Set the frame width and height
    frame_width = int(vid.get(CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video file
    out = VideoWriter(output_path, VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    # Read and process the frames one by one
    previous_frame = None

    print("Usuwanie zduplikowanych klatek...")
    pbar = tqdm(range(vidlength))
    duplicated_num = 0
    for _ in pbar:
        # Read the next frame
        success, frame = vid.read()

        # If we reached the end of the video, break the loop
        if not success:
            break

        # If the current frame is not a duplicate, write it to the output video
        if previous_frame is None or PSNR(frame, previous_frame) < 40.:
            out.write(frame)
        else:
            duplicated_num += 1
            pbar.set_description(f"Ilość klatek-duplikatów: {duplicated_num}")

        # Update the previous frame
        previous_frame = frame
    print('Zakończono usuwanie zduplikowanych klatek.')


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    p  = Path(application_path)
    video_path = next(p.glob('*.mp4'))
    toRemove = input('Czy usuwać zduplikowane klatki? Wpisz "t" lub "n": \n')
    if toRemove == 't':
        bez_duplikatow_video_path = str(video_path.with_name('bez_duplikatow_' + video_path.name))
        remove_duplicate_frames(str(video_path), bez_duplikatow_video_path)
        video_path = bez_duplikatow_video_path
    video_path = str(video_path)
    player = VideoPlayer(video_path)
    while(player.cap.isOpened()):
        player.run()
