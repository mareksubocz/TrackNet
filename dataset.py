import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import argparse
import pandas as pd
import numpy as np
import abc
import sys

# if "ipykernel" in sys.modules:  # executed in a jupyter notebook
#     from tqdm.notebook import tqdm
# else:
#     from tqdm import tqdm
from tqdm import tqdm

from pathlib import Path
import cv2 as cv


class GenericDataset(Dataset):
    def __init__(self, opt):
        """Generic Dataset class, allows for dataset creation without specyfing the type of the dataset (images/videos).

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """
        self.opt = opt
        self.image_size = opt.image_size
        self.base_path = Path(opt.dataset)
        self.csvs_dir = self.base_path / opt.csvs_dir
        self.sequence_starters = {}
        self.sequence_length = opt.sequence_length
        self.one_output_frame = opt.one_output_frame
        self.grayscale = opt.grayscale


        print('Calculating sequence starters...')
        for csv_path in tqdm(list(self.csvs_dir.glob("*.csv"))):
            self.sequence_starters[csv_path.stem] = []
            # reset prev_nums
            prev_nums = [-self.sequence_length] * (self.sequence_length - 1)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # checking if the sequence is consecutive
                for i, prev_num in enumerate(prev_nums, start=1):
                    if prev_num != row["frame_num"] - i:
                        break
                else:
                    self.sequence_starters[csv_path.stem].append(
                        int(row["frame_num"]) - (self.sequence_length - 1))
                    # reset prev_nums
                    if not opt.include_dups:
                        prev_num = [-self.sequence_length] * (self.sequence_length - 1)
                    continue
                prev_nums = [row["frame_num"]] + prev_nums[:-1]
        print()

    def __len__(self):
        return sum(len(s) for s in self.sequence_starters.values())

    def __getitem__(self, idx):
        rel_idx = idx
        img_name = ""

        for curr_name in list(self.csvs_dir.glob("*.csv")):
            starters = self.sequence_starters[curr_name.stem]
            # next folder
            if rel_idx >= len(starters):
                rel_idx -= len(starters)
                continue
            # found the folder
            img_name = curr_name.stem
            rel_idx = starters[rel_idx]
            break

        df = pd.read_csv((self.csvs_dir / img_name).with_suffix(".csv"))
        if 'w' not in df.columns:
            df['w'] = 50
        if 'h' not in df.columns:
            df['h'] = 50
        if self.one_output_frame:
            df = df.loc[df['frame_num'] == rel_idx + self.sequence_length // 2].iloc[0]
            heatmaps = self.generate_heatmap_2(df["x"], df["y"], df["w"], df["h"])
            # heatmaps = self.generate_heatmap(df["x"], df["y"], 100, 50)

            heatmaps = np.expand_dims(heatmaps, axis=0)
        else:
            df = df.loc[df['frame_num'].isin(range(rel_idx, rel_idx + self.sequence_length))]
            heatmaps = []
            for _, row in df.iterrows():
                heatmaps.append(self.generate_heatmap_2(row["x"], row["y"], row["w"], row["h"]))
                # heatmaps.append(self.generate_heatmap(row["x"], row["y"], 100, 50))
            heatmaps = np.stack(heatmaps, axis=0)

        heatmaps = torch.tensor(heatmaps, requires_grad=False, dtype=torch.float32)
        images = self.get_images(img_name, int(rel_idx))

        if self.grayscale:
            X_grayscale = [torchvision.transforms.functional.rgb_to_grayscale(images[3*i:3*(i+1),:,:]) for i in range(self.sequence_length)]
            images = torch.cat(X_grayscale, axis=0)


        return images, heatmaps

    def get_images(self, img_name: str | Path, rel_idx: int) -> torch.Tensor:
        """A virtual method for implementation in Images/Videos dataset classes.

        Implement this in a subclass to return a sequence of images in a form of a torch tensor.
        """
        raise NotImplementedError

    def generate_heatmap(self, center_x, center_y, variance, size):
        x = int(center_x * self.image_size[1])
        y = int(center_y * self.image_size[0])
        x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * variance))

        image = np.zeros(self.image_size)
        image = np.pad(image, size)
        image[y:y + (size*2) + 1, x:x + (size*2) + 1] = g
        image = image[size:-size, size:-size]
        return image


    def generate_heatmap_2(self, center_x, center_y, width, height):
        """ Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.

        source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        """
        x = np.arange(0, self.image_size[1], 1, float)
        y = np.arange(0, self.image_size[0], 1, float)[:,np.newaxis]

        x0 = self.image_size[1]*center_x
        y0 = self.image_size[0]*center_y
        width = self.image_size[1]*width
        height = self.image_size[0]*height

        image = np.exp(-4*np.log(2) * ((x-x0)**2/width**2 + (y-y0)**2/height**2))
        return image



    @staticmethod
    def from_dir(opt):
        """Generate a dataset of adequate type given root directory.

        Args:
            root_dir (str | Path): Path to the dataset folder.
            images_folder (str, optional): Name of the folder containing images.
            videos_folder (str, optional): Name of the folder containing videos.

        Returns:
            ImagesDataset | VideosDataset: Dataset of adequate type, if both videos and images folders are present, ImagesDataset is returned.
        """
        base_path = Path(opt.dataset)
        if (base_path / opt.images_dir).is_dir():
            return ImagesDataset(opt)
        elif (base_path / opt.videos_dir).is_dir():
            return VideosDataset(opt)
        else:
            raise Exception(f"No '{opt.images_dir}' or '{opt.videos_dir}' folder found in dataset.")



class ImagesDataset(GenericDataset):
    def __init__(self, opt):
        """Pytorch dataset utilizing videos cut into frames.
        Images are divided into folders named after the video they were taken from.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            images_folder (str, optional): Name of the folder containing images folders.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """

        super().__init__(opt)
        self.images_folder = self.base_path / opt.images_dir

    def get_images(self, img_dir, rel_idx):
        images = []
        for i in range(self.sequence_length):
            image_path = self.images_folder / img_dir / str(rel_idx+i)
            img = torch.tensor([], requires_grad=False)
            if image_path.with_suffix(".png").is_file():
                img = torchvision.io.read_image(str(image_path.with_suffix(".png")))
            elif image_path.with_suffix(".jpeg").is_file():
                img = torchvision.io.read_image(image_path.with_suffix(".jpeg"))
            elif image_path.with_suffix(".jpg").is_file():
                img = torchvision.io.read_image(str(image_path.with_suffix(".jpg")))
            else:
                raise Exception(f"Image {rel_idx} in folder {img_dir} not found")

            img = torchvision.transforms.functional.resize(img, self.image_size)
            img = img.type(torch.float32)
            img *= 1 / 255
            images.append(img)

        return torch.concatenate(images)


# TODO: random access is slow, use IterableDataset and VideoReader for faster reading?
class VideosDataset(GenericDataset):
    def __init__(self, opt):
        """Pytorch dataset utilizing videos in .mp4 format.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            videos_folder (str, optional): Name of the folder containing videos.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """

        super().__init__(opt)
        self.videos_folder = self.base_path / opt.videos_dir

    def get_images(self, img_dir, rel_idx):
        cap = cv.VideoCapture(
            str((self.videos_folder / img_dir).with_suffix(".mp4")))
        cap.set(cv.CAP_PROP_POS_FRAMES, rel_idx)
        images = []
        for _ in range(self.sequence_length):
            _, img = cap.read()
            # using opencv the image will have channels as last dimension
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            img = torchvision.transforms.functional.resize(
                img, self.image_size)
            img = img.type(torch.float32)
            img *= 1 / 255
            images.append(img)
        cap.release()

        return torch.concatenate(images)

    def to_images_dataset(self, images_folder="images"):
        """Convert VideosDataset to ImagesDataset for faster loading.
        Requires splitting videos into frames and saving the frames as .png.

        Args:
            images_folder (str, optional): Name of the folder containing images folders.

        Returns:
            ImagesDataset: Dataset with images instead of videos.
        """
        images_dir = self.base_path / images_folder
        images_dir.mkdir(exist_ok=True)
        videos_list = list(self.videos_folder.glob("*.mp4"))
        for video_num, video_path in enumerate(videos_list):
            (images_dir / video_path.stem).mkdir(exist_ok=True)
            cap = cv.VideoCapture(str(video_path))
            frame_num = 0
            print(f"--({video_num+1}/{len(videos_list)})-- splitting {video_path.name}")
            for _ in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
                success, frame = cap.read()
                if success:
                    cv.imwrite(
                        str((images_dir / video_path.name / str(frame_num)).with_suffix(".png")),
                        frame)
                else:
                    break
                frame_num += 1
            cap.release()

        return ImagesDataset(self.opt)


# def makeGaussian(image_size, size, variance, center):
#     """_summary_
#
#     Args:
#         image_size (_type_): _description_
#         size (_type_): _description_
#         variance (int): To have the ball as value > 0.5, variance should be ball_radius * 8
#         center (_type_): _description_
#
#     Returns:
#         _type_: _description_
#     """
#     x = int(center[1] * image_size[1])
#     y = int(center[0] * image_size[0])
#     x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
#     g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * variance))
#
#     image = np.zeros(image_size)
#     image = np.pad(image, size)
#     image[y:y + (size*2) + 1, x:x + (size*2) + 1] = g
#     image = image[size:-size, size:-size]
#     return image


def makeGaussian(size = (1200, 600), fwhm = (100, 50), center=(0.5, 0.5)):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    """

    x = np.arange(0, size[0], 1, float)
    # y = x[:,np.newaxis]
    y = np.arange(0, size[1], 1, float)[:,np.newaxis]

    x0 = size[0]*center[0]
    y0 = size[1]*center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2/fwhm[0]**2 + (y-y0)**2/fwhm[1]**2))


if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    # plt.imshow(makeGaussian(center=(0.1, 0.4)), interpolation='nearest', vmin=0, vmax=1)
    # plt.show()
    #
    # exit(0)
    opt = argparse.Namespace()
    opt.dataset = "./example_datasets/video_dataset/"
    opt.images_dir = "images"
    opt.videos_dir = "videos"
    opt.csvs_dir = "csvs"
    opt.sequence_length = 3
    opt.one_output_frame = False
    opt.grayscale = False
    opt.image_size = (1024, 512)
    opt.include_dups = True

    dataset = GenericDataset.from_dir(opt)
    # dataset = dataset.to_images_dataset()
    dl = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (x, y) in enumerate(dl):
        print(f"{i}: ", x.shape, y.shape)


    # generate heatmap
    x = 0.4
    y = 0.7
    variance = 3
    size = 3
    image_size = (512, 1024)

    x = int(x * image_size[1])
    y = int(y * image_size[0])
    x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * variance))

    image = np.zeros(image_size)
    image = np.pad(image, size)
    image[y:2 * size + y + 1, x:2 * size + x + 1] = g
    image = image[size:-size, size:-size]
    print(image.shape)
    print(image_size)
    save_image(torch.from_numpy(image), "test.png")
