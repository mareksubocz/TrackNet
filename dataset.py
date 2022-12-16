import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, root_dir, image_size=(512, 1024), force_recalculate_heatmaps=False, csvs_folder="csvs", heatmaps_folder="heatmaps", sequence_length=3):
        """Generic Dataset class, allows for dataset creation without specyfing the type of the dataset (images/videos).

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            force_recalculate_heatmaps (bool, optional): Should heatmaps be recalculated even if they are already present as a file.
            csvs_folder (str, optional): Name of the folder containing csvs.
            heatmaps_folder (str, optional): Name of the folder containing heatmaps.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """
        self.image_size = image_size
        self.base_path = Path(root_dir)
        self.csvs_dir = self.base_path / csvs_folder
        self.csvs_paths = self.csvs_dir.glob("*.csv")
        self.heatmaps_dir = self.base_path / heatmaps_folder
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.sequence_starters = {}
        self.sequence_length = sequence_length

        # generate heatmaps folder and content
        print("Generating heatmaps...")
        for csv_path in tqdm(list(self.csvs_paths)):
            self.sequence_starters[csv_path.stem] = []
            prev_nums = [-self.sequence_length] * (self.sequence_length - 1)
            curr_path = self.heatmaps_dir / csv_path.stem
            curr_path.mkdir(exist_ok=True)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # TODO: utilize visibility info
                if (force_recalculate_heatmaps or not (curr_path / str(row["num"])).with_suffix(".npy").is_file()):
                    heatmap = self.generate_heatmap(row["x"], row["y"], 100,
                                                    50)
                    np.save((curr_path / str(row["num"])).with_suffix(".npy"), heatmap)

                # checking if the sequence is consecutive
                for i, prev_num in enumerate(prev_nums, start=1):
                    if prev_num != row["num"] - i:
                        break
                else:
                    self.sequence_starters[csv_path.stem].append(
                        row["num"] - (self.sequence_length - 1))
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
        heatmaps = []
        for i in range(self.sequence_length):
            heatmaps.append(
                np.expand_dims(
                    np.load((self.heatmaps_dir / img_name / str(rel_idx + i)).with_suffix(".npy")),
                    axis=0,
                ))
        heatmaps = torch.tensor(np.concatenate(heatmaps), requires_grad=False, dtype=torch.float32)

        images = self.get_images(img_name, rel_idx)

        return images, heatmaps

    def get_images(self, img_name: str | Path, rel_idx: int) -> torch.Tensor:
        """A virtual method for implementation in Images/Videos dataset classes.

        Implement this in a subclass to return a sequence of images in a form of a torch tensor.
        """
        raise NotImplementedError

    def generate_heatmap(self, x, y, variance, size):
        x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * variance))

        image = np.zeros(self.image_size)
        image = np.pad(image, size * 2)
        # FIXME: ValueError: could not broadcast input array from shape (3,3) into shape (3,0)
        image[y:2 * size + y + 1, x:2 * size + x + 1] = g
        image = image[variance:-variance, variance:-variance]
        return image

    @staticmethod
    def from_dir(root_dir, images_folder="images", videos_folder="videos"):
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
            return ImagesDataset(root_dir, images_folder=images_folder)
        elif (base_path / videos_folder).is_dir():
            return VideosDataset(
                root_dir,
                videos_folder=videos_folder,
                force_recalculate_heatmaps=True,
            )
        else:
            assert f"No '{images_folder}' or '{videos_folder}' folder found in dataset."


class ImagesDataset(GenericDataset):
    def __init__(self, root_dir, image_size=(512, 1024), force_recalculate_heatmaps=True, csvs_folder="csvs", heatmaps_folder="heatmaps", images_folder="images", sequence_length=3):
        """Pytorch dataset utilizing videos cut into frames. 
        Images are divided into folders named after the video they were taken from.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            force_recalculate_heatmaps (bool, optional): Should heatmaps be recalculated even if they are already present as a file.
            csvs_folder (str, optional): Name of the folder containing csvs.
            heatmaps_folder (str, optional): Name of the folder containing heatmaps.
            images_folder (str, optional): Name of the folder containing images folders.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """

        super().__init__(
            root_dir,
            image_size=image_size,
            force_recalculate_heatmaps=force_recalculate_heatmaps,
            csvs_folder=csvs_folder,
            heatmaps_folder=heatmaps_folder,
            sequence_length=sequence_length,
        )
        self.images_folder = self.base_path / images_folder

    def get_images(self, img_dir, rel_idx):
        images = []
        for _ in range(self.sequence_length):
            image_path = self.images_folder / img_dir / str(rel_idx)
            img = torch.tensor([], requires_grad=False)
            if image_path.with_suffix(".png").is_file():
                img = torchvision.io.read_image(str(image_path.with_suffix(".png")))
            elif image_path.with_suffix(".jpeg").is_file():
                img = torchvision.io.read_image(
                    image_path.with_suffix(".jpeg"))
            elif image_path.with_suffix(".jpg").is_file():
                img = torchvision.io.read_image(str(image_path.with_suffix(".jpg")))
            else:
                assert f"No image {rel_idx} in folder {img_dir} found."

            img = torchvision.transforms.functional.resize(
                img, self.image_size)
            img = img.type(torch.float32)
            img *= 1 / 255
            images.append(img)

        return torch.concatenate(images)


# TODO: random access is slow, use IterableDataset and VideoReader for faster reading?
class VideosDataset(GenericDataset):
    def __init__(self, root_dir, image_size=(512, 1024), force_recalculate_heatmaps=True, csvs_folder="csvs", heatmaps_folder="heatmaps", videos_folder="videos", sequence_length=3):
        """Pytorch dataset utilizing videos in .mp4 format.

        Args:
            root_dir (string): Path to the dataset folder.
            image_size (tuple[int], optional): Size that the images will be resized to.
            force_recalculate_heatmaps (bool, optional): Should heatmaps be recalculated even if they are already present as a file.
            csvs_folder (str, optional): Name of the folder containing csvs.
            heatmaps_folder (str, optional): Name of the folder containing heatmaps.
            videos_folder (str, optional): Name of the folder containing videos.
            sequence_length (int, optional): Length of the image sequence used for ball detection.
        """
        super().__init__(
            root_dir,
            image_size=image_size,
            force_recalculate_heatmaps=force_recalculate_heatmaps,
            csvs_folder=csvs_folder,
            heatmaps_folder=heatmaps_folder,
            sequence_length=sequence_length,
        )
        self.videos_folder = self.base_path / videos_folder

    def get_images(self, img_dir, rel_idx):
        cap = cv.VideoCapture(
            str((self.videos_folder / img_dir).with_suffix(".mp4")))
        cap.set(cv.CAP_PROP_POS_FRAMES, rel_idx)
        images = []
        for _ in range(self.sequence_length):
            ret, img = cap.read()
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

        return ImagesDataset(self.base_path, images_folder=images_folder)


if __name__ == "__main__":
    dataset = ImagesDataset("./dataset/")
    # dataset = dataset.to_images_dataset()
    dl = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (x, y) in enumerate(dl):
        print(f"{i}: ", x.shape, y.shape)
