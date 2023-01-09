#manage imports
import cv2 as cv
import numpy as np
from pathlib import Path
from dataset import GenericDataset

## check bitcoin prices
def visualize_heatmaps(path_to_heatmaps_folder, path_to_video = None):
    path_to_heatmaps_folder = Path(path_to_heatmaps_folder)
    images_paths = sorted(path_to_heatmaps_folder.glob('*.npy'))
    if path_to_video:
        cap = cv.VideoCapture(path_to_video)

    # show loop
    for path in images_paths:
        i = int(path.stem)
        img = np.load(str(path))
        _, thresh1 = cv.threshold(img,0.1,1,cv.THRESH_BINARY)

        if path_to_video:
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (img.shape[1], img.shape[0]))
            cv.imshow('frame', frame)
            cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 3)

            # res = np.array([f * img for f in [frame[:,:,0], frame[:,:,1], frame[:,:,2]]]).transpose(1,2,0)
            res = cv.bitwise_and(frame, frame, mask=np.uint8(thresh1))
            cv.imshow('heatmaps*frame', res)
            cv.setWindowProperty('heatmaps*frame', cv.WND_PROP_TOPMOST, 1)
        cv.imshow('heatmaps', img)
        cv.setWindowProperty('heatmaps', cv.WND_PROP_TOPMOST, 2)
        if cv.waitKey(100000) & 0xFF == ord('q'):
            break
    
    
    
if __name__ == '__main__':
    dataset = GenericDataset.from_dir('./dataset_no_dups', force_recalculate_heatmaps=False)
    visualize_heatmaps('dataset_no_dups/heatmaps/NoDups1/', path_to_video='dataset_no_dups/videos/NoDups1.mp4')