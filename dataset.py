import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import pandas as pd
import numpy as np
import sys

if "ipykernel" in sys.modules:  # executed in a jupyter notebook
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pathlib import Path
import cv2 as cv


class GenericDataset(Dataset):
    def __init__(self, base_path, image_size=(360, 640), csvs_folder="csvs", sequence_length=3, one_output_frame=False, grayscale=False):
        """Generic Dataset class, allows for dataset creation without specyfing the type of the dataset (images/videos).

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """
        self.image_size = image_size
        self.base_path = Path(base_path)
        self.csvs_dir = self.base_path / csvs_folder
        self.sequence_starters = {}
        self.sequence_length = sequence_length
        self.one_output_frame = one_output_frame
        self.grayscale = grayscale


        print('Calculating sequence starters...')
        for csv_path in tqdm(list(self.csvs_dir.glob("*.csv"))):
            self.sequence_starters[csv_path.stem] = []
            prev_nums = [-self.sequence_length] * (self.sequence_length - 1)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # checking if the sequence is consecutive
                for i, prev_num in enumerate(prev_nums, start=1):
                    if prev_num != row["num"] - i:
                        break
                else:
                    self.sequence_starters[csv_path.stem].append(
                        int(row["num"]) - (self.sequence_length - 1))
                prev_nums = [row["num"]] + prev_nums[:-1]
        print()

    def __len__(self):
        return sum(len(v) for v in self.sequence_starters.values())

    def __getitem__(self, idx):
        rel_idx = idx
        img_name = ""

        for curr_name, starters in self.sequence_starters.items():
            # next folder
            if rel_idx >= len(starters):
                rel_idx -= len(starters)
                continue
            # found the folder
            img_name = curr_name
            rel_idx = starters[rel_idx]
            break

        df = pd.read_csv((self.csvs_dir / img_name).with_suffix(".csv"))
        if self.one_output_frame:
            df = df.loc[df['num'] == rel_idx + self.sequence_length // 2].iloc[0]
            heatmaps = self.generate_heatmap(df["x"], df["y"], 100, 50)
        else:
            df = df.loc[df['num'].isin(range(rel_idx, rel_idx + self.sequence_length))]
            heatmaps = []
            for _, row in df.iterrows():
                heatmaps.append(self.generate_heatmap(row["x"], row["y"], 100, 50))
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

    # TODO: utilize visibility info
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

    @staticmethod
    def from_dir(root_dir, images_folder="images", videos_folder="videos", one_output_frame=False, grayscale=False):
        """Generate a dataset of adequate type given root directory.

        Args:
            root_dir (str | Path): Path to the dataset folder.
            images_folder (str, optional): Name of the folder containing images.
            videos_folder (str, optional): Name of the folder containing videos.

        Returns:
            ImagesDataset | VideosDataset: Dataset of adequate type, if both videos and images folders are present, ImagesDataset is returned.
        """
        base_path = Path(root_dir)
        if (base_path / images_folder).is_dir():
            return ImagesDataset(root_dir, images_folder=images_folder, one_output_frame=one_output_frame)
        elif (base_path / videos_folder).is_dir():
            return VideosDataset(
                root_dir,
                videos_folder=videos_folder,
                one_output_frame=one_output_frame,
                grayscale=grayscale
            )
        else:
            raise Exception(f"No '{images_folder}' or '{videos_folder}' folder found in dataset.")



class ImagesDataset(GenericDataset):
    def __init__(self, root_dir, image_size=(512, 1024), csvs_folder="csvs", images_folder="images", sequence_length=3, one_output_frame=False, grayscale=False):
        """Pytorch dataset utilizing videos cut into frames. 
        Images are divided into folders named after the video they were taken from.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            images_folder (str, optional): Name of the folder containing images folders.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """

        super().__init__(
            root_dir,
            image_size=image_size,
            csvs_folder=csvs_folder,
            sequence_length=sequence_length,
            one_output_frame=one_output_frame,
            grayscale=grayscale
        )
        self.images_folder = self.base_path / images_folder

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
    def __init__(self, root_dir, image_size=(512, 1024), csvs_folder="csvs", videos_folder="videos", sequence_length=3, one_output_frame=False, grayscale=False):
        """Pytorch dataset utilizing videos in .mp4 format.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            csvs_folder (str, optional): Name of the folder containing csvs.
            videos_folder (str, optional): Name of the folder containing videos.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """
        
        super().__init__(
            root_dir,
            image_size=image_size,
            csvs_folder=csvs_folder,
            sequence_length=sequence_length,
            one_output_frame=one_output_frame,
            grayscale=grayscale
        )
        self.videos_folder = self.base_path / videos_folder

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
            (images_dir / video_path.name).mkdir(exist_ok=True)
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

        return ImagesDataset(self.base_path, images_folder=images_folder, one_output_frame=self.one_output_frame, grayscale=self.grayscale)


if __name__ == "__main__":
    dataset = GenericDataset.from_dir("./dataset/")
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