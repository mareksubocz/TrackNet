import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
import cv2 as cv
import os


#TODO: test this :D
class GenericDataset(Dataset):
    def __init__(self, 
                 base_path, 
                 image_size = (1200,600),
                 force_recalculate_heatmaps=False,
                 csvs_folder = 'csvs',
                 heatmaps_folder = 'heatmaps',
                 sequence_length = 3):
        self.image_size = image_size
        self.base_path = Path(base_path)
        self.csvs_dir = self.base_path.joinpath(csvs_folder)
        self.csvs_paths = self.csvs_dir.glob('*.csv')
        self.heatmaps_dir = self.base_path.joinpath(heatmaps_folder)
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.sequence_starters = {}
        self.sequence_length = sequence_length

        # generate heatmaps folder and content
        print('Generating heatmaps...')
        for csv_path in tqdm(self.csvs_paths):
            self.sequence_starters[csv_path.stem] = []
            prev_nums = [-self.sequence_length]*(self.sequence_length-1)
            curr_path = self.heatmaps_dir.joinpath(csv_path.stem)
            curr_path.mkdir(exist_ok=True)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                #TODO: utilize visibility info
                if force_recalculate_heatmaps or not curr_path.joinpath(row['num']).with_suffix('.npy').is_file():
                    heatmap = self.generate_heatmap(row['x'], row['y'], 10, 1)
                    np.save(curr_path.joinpath(row['num']).with_suffix('.npy'), heatmap)

                # checking if the sequence is consecutive
                for i, prev_num in enumerate(prev_nums, start=1):
                    if prev_num != row['num']-i:
                        break
                else:
                    self.sequence_starters[csv_path.stem].append(row['num'] - (self.sequence_length-1))
                prev_nums = [row['num']] + prev_nums[:-1]


    def __len__(self):
        return sum(len(v) for v in self.sequence_starters.values())


    def __getitem__(self, idx):
        rel_idx = idx
        img_dir = ''

        for curr_dir, starters in self.sequence_starters.items():
            # next folder
            if rel_idx >= len(starters):
                rel_idx -= len(starters)
                continue
            # found the folder
            img_dir = curr_dir
            break
                
        heatmaps = []
        #FIXME: check if channels are the first dimension
        for i in range(self.sequence_length):
            heatmaps.append(np.load(
                self.csvs_dir.joinpath(img_dir).joinpath(rel_idx+i).with_suffix('.npy')
            ))
        heatmaps = torch.tensor(np.concatenate(heatmaps), requires_grad=False)
        
        images = self.get_images(img_dir, rel_idx)
        
        return images, torch.tensor(heatmaps, requires_grad=False)
    

    def get_images(self, img_dir, rel_idx):
        """A virtual method for implementation in Images/Videos dataset classes.

        Args:
            img_dir (string): name of the image folder / movie without suffix
            rel_idx (int): number of the sequence-starting frame

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    
    def generate_heatmap(self, x, y, variance, size):
        x_grid, y_grid = np.mgrid[-size:size+1, -size:size+1]
        g = np.exp(-(x_grid**2+y_grid**2)/float(2*variance))

        image = np.zeros(self.image_size)
        image = np.pad(image, size*2)
        image[y:2*size + y + 1, x:2*size + x + 1] = g
        image = image[variance:-variance, variance:-variance]
        return image
    
    
    @staticmethod
    def from_folder(dataset_path, images_folder='images', videos_folder='videos'):
        base_path = Path(dataset_path)
        if base_path.joinpath(images_folder).is_dir():
            return ImagesDataset(dataset_path, images_folder=images_folder)
        elif base_path.joinpath(videos_folder).is_dir():
            return VideosDataset(dataset_path, videos_folder=videos_folder)
        else:
            assert(f"No '{images_folder}' or '{videos_folder}' folder found in dataset.")


class ImagesDataset(GenericDataset):
    def __init__(self, 
                 base_path, 
                 image_size = (1200,600),
                 force_recalculate_heatmaps=False,
                 csvs_folder = 'csvs',
                 heatmaps_folder = 'heatmaps',
                 images_folder = 'images',
                 sequence_length = 3):
        super().__init__(base_path,
                         image_size=image_size,
                         force_recalculate_heatmaps=force_recalculate_heatmaps,
                         csvs_folder=csvs_folder,
                         heatmaps_folder=heatmaps_folder,
                         sequence_length=sequence_length)
        self.images_folder = self.base_path.joinpath(images_folder)
        
    
    def get_images(self, img_dir, rel_idx):
        images = []
        for _ in range(self.sequence_length):
            image_path = self.images_folder.joinpath(img_dir).joinpath(rel_idx)
            img = torch.tensor(requires_grad=False)
            if image_path.with_suffix('.png').is_file():
                img = torchvision.io.read_image(
                    image_path.with_suffix('.png')
                )
            elif image_path.with_suffix('.jpeg').is_file():
                img = torchvision.io.read_image(
                    image_path.with_suffix('.jpeg')
                )
            elif image_path.with_suffix('.jpg').is_file():
                img = torchvision.io.read_image(
                    image_path.with_suffix('.jpg')
                )
            else:
                assert(f'No image {rel_idx} in folder {img_dir} found.')
                
            img = torchvision.transforms.functional.resize(img, self.image_size)
            img = img.type(torch.FloatTensor)
            img *= 1/255
            images.append(img)

        return torch.concatenate(images)


#TODO: random access is slow, use IterableDataset and VideoReader for faster reading?
class VideosDataset(GenericDataset):
    def __init__(self, 
                 dataset_path, 
                 image_size = (1200,600),
                 force_recalculate_heatmaps=False,
                 csvs_folder = 'csvs',
                 heatmaps_folder = 'heatmaps',
                 videos_folder = 'videos',
                 sequence_length = 3):
        super().__init__(dataset_path,
                         image_size=image_size,
                         force_recalculate_heatmaps=force_recalculate_heatmaps,
                         csvs_folder=csvs_folder,
                         heatmaps_folder=heatmaps_folder,
                         sequence_length=sequence_length)
        self.videos_folder = self.base_path.joinpath(videos_folder)
        
    
    def get_images(self, img_dir, rel_idx):
        cap = cv.VideoCapture(self.base_path.joinpath(img_dir).with_suffix('.mp4'))
        cap.set(cv.CAP_PROP_POS_FRAMES, rel_idx)
        images = []
        for _ in range(self.sequence_length):
            ret, img = cap.read()
            # using opencv the image will have channels as last dimension
            img = img.transpose(2,0,1)
            img = torch.from_numpy(img)
            img = img.type(torch.FloatTensor)
            img *= 1/255
            images.append(img)
        cap.release()

        return torch.concatenate(images)
    
    def to_images_dataset(self, images_folder = 'images'):
        images_dir = self.base_path.joinpath(images_folder)
        images_dir.mkdir(exist_ok=True)
        for video_path in self.videos_folder.glob('*.mp4'):
            images_dir.joinpath(video_path).mkdir(exist_ok=True)
            cap = cv.VideoCapture(video_path)
            frame_num = 0
            while True:
                success, frame = cap.read()
                if success:
                    cv.imwrite(
                        images_dir.joinpath(frame_num).with_suffix('.png'),
                        frame)
                else:
                    break
                frame_num += 1
            cap.release()

        return ImagesDataset(self.base_path, images_folder=images_folder)


############################ HEATMAP GENERATION ############################

def insert_at_pos(arrs, kernel, xs, ys):
    size = kernel.shape[0]
    for arr, x, y in zip(arrs, xs, ys):
        print(arr[x-size:x+size+1,  y-size:y+size+1].shape)
        arr[x:x+size+1,  y-size:y+size+1] = kernel

    return arrs


def generate_heatmap_from_df(height:int, width:int, df, variance:float=200, size: int=100):
    """
    Generate heatmap indicating the ball position using Gaussian Distibution.

    Parameters
    ----------
    height : int
        height of the input image
    width : int
        width of the input image
    df : pd.DataFrame
        dataframe with xs and ys of the ball location
    variance : float
        variance of the Gaussian Distibution
    size : int
        size of the field with Gaussian Distribution
    """

    df['x'] = (df['x']*1200).astype('int')
    df['y'] = (df['y']*600).astype('int')

    # gaussian kernel
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))

    images = np.zeros((len(df), width + size*2, height + size*2))
    # v_insert = np.vectorize(insert_at_pos)

    print(df)
    insert_at_pos(images, g, df['x'], df['y'])
    images = images[:,size:-size, size:-size]
    return images


def generate_heatmap(height:int, width:int, xb:int, yb:int, variance:float=200, size: int=100):
    """
    Generate heatmap indicating the ball position using Gaussian Distibution.

    Parameters
    ----------
    height : int
        height of the input image
    width : int
        width of the input image
    xb : int
        x coordinate of the ball
    yb : int
        y coordinate of the ball
    variance : float
        variance of the Gaussian Distibution
    size : int
        size of the field with Gaussian Distribution
    """

    # gaussian kernel
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))

    image = np.zeros((width, height))
    image = np.pad(image, size*2)
    image[yb:2*size + yb + 1, xb:2*size + xb + 1] = g
    image = image[variance:-variance, variance:-variance]
    return image


if __name__ == "__main__":
    df = pd.read_csv('../Beaver/BeaverShortTrackNet/labels/cut+ZAKSA-Bel.csv')
    # res = generate_heatmap_from_df(1200, 600, df)
    df['x'] = (df['x']*1200).astype('int')
    df['y'] = (df['y']*600).astype('int')
    res = []
    resColor = []
    cap = cv.VideoCapture('../Beaver/BeaverShortTrackNet/videos/cut+ZAKSA-Bel.mp4')
    for x, y, frame_num in zip(df['x'], df['y'], df['frame_num']):
        res.append(generate_heatmap(1200, 600, x, y, 200, 100))
        cap.set(1, frame_num)
        ret, frame = cap.read()
        frame = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (1200, 600))
        frame = np.where(res[-1] < 0.1, 0, frame)
        resColor.append(frame)
    import imageio
    imageio.mimwrite('res.mp4', res, fps=30)
    imageio.mimwrite('resColor.mp4', resColor, fps=30)
    # import matplotlib.pyplot as plt
    # plt.imshow(hm*255)
    # plt.show()
