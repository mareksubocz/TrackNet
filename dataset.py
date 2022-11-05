import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.DataFrame(pd.read_csv(csv_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = {'text': row[0], 'number': row[1]}
        return sample

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
    image = np.pad(image, size)
    image[xb:2*size + xb + 1, yb:2*size + yb + 1] = g
    image = image[variance:-variance, variance:-variance]
    return image

if __name__ == "__main__":
    hm = generate_heatmap(1200, 600, 200, 300, 200, 100)
    import matplotlib.pyplot as plt
    plt.imshow(hm*255)
    plt.show()
