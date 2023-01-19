from argparse import ArgumentParser
from TrackNet import TrackNet
import torchvision
import torch
import cv2

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
    model.load(opt.weights, device = device)
    model.eval()
    
    cap = cv2.VideoCapture(opt.video)
    prev_frame_1 = None
    prev_frame_2 = None
    while cap.isOpened():
        ret,frame = cap.read()
        cv2.imshow('original', frame)
        frame = torch.tensor(frame).permute(2, 0, 1).float().to(device) / 255
        frame = torchvision.transforms.functional.resize(frame, opt.image_size)


        if prev_frame_1 is not None and prev_frame_2 is not None:
            frames = torch.cat([prev_frame_2, prev_frame_1, frame], dim=0).unsqueeze(0)
            pred = model(frames)
            pred = pred[0,0,:,:].detach().cpu().numpy()
            cv2.imshow('prediction', pred)
        prev_frame_2 = prev_frame_1
        prev_frame_1 = frame

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows