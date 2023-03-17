from argparse import ArgumentParser
from TrackNet import TrackNet
import numpy as np
import torchvision
import torch
import cv2 as cv


def get_ball_position(img, original_img_=None):
    ret, thresh = cv.threshold(img, 0.9, 1, 0)
    thresh = cv.convertScaleAbs(thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # numpy image
    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv.contourArea)

        if original_img_ is not None:
            # the contours are drawn here
            cv.drawContours(original_img_, [c], -1, 255, 3)
            cv.drawContours(img, [c], -1, 255, 3)

        x,y,w,h = cv.boundingRect(c)
        return x, y, w, h


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('video', type=str, default='video.mp4', help='Path to video.')
    parser.add_argument('--save_path', type=str, default='predicted.mp4', help='Path to result video.')
    parser.add_argument('--weights', type=str, default='weights', help='Path to trained model weights.')
    parser.add_argument('--sequence_length', type=int, default=3, help='Length of the images sequence used as X.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help='Size of the images used for training (y, x).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--one_output_frame', action='store_true', help='Demand only one output frame instead of three.')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images instead of RGB.')
    parser.add_argument('--visualize', action='store_true', help='Display the predictions in real time.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    opt.dropout = 0
    device = torch.device(opt.device)
    model = TrackNet(opt).to(device)
    model.load(opt.weights, device = opt.device)
    model.eval()

    cap = cv.VideoCapture(opt.video)
    prev_frame_1 = None
    prev_frame_2 = None
    while cap.isOpened():
        ret,frame = cap.read()
        frame_torch = torch.tensor(frame).permute(2, 0, 1).float().to(device) / 255
        frame_torch = torchvision.transforms.functional.resize(frame_torch, opt.image_size)


        if prev_frame_1 is not None and prev_frame_2 is not None:
            frames = torch.cat([prev_frame_2, prev_frame_1, frame_torch], dim=0).unsqueeze(0)
            pred = model(frames)
            pred = pred[0,:,:,:].detach().cpu().numpy()
            for i in range(pred.shape[0]):
                pred_cut = pred[i,:,:]
                frame_resized = cv.resize(frame, pred_cut.shape[::-1], interpolation = cv.INTER_AREA)
                get_ball_position(pred_cut, original_img_=frame_resized)
                cv.imshow('prediction', pred_cut)
                cv.imshow('original', frame_resized)
        prev_frame_2 = prev_frame_1
        prev_frame_1 = frame_torch

        if cv.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows() # destroy all opened windows
