import torch
from torch import nn

class TrackNet(nn.Module):

    def _make_convolution_layer(self, v, num):
        layers = [
            nn.Conv2d(9, v, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=v)
        ]

        return nn.Sequential(
            *(layers*num)
        )


    def __init__(self):
        super(TrackNet, self).__init__()

        # VGG16
        self.vgg_conv1 = self._make_convolution_layer(64, 2)
        self.vgg_maxpool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv2 = self._make_convolution_layer(128, 2)
        self.vgg_maxpool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv3 = self._make_convolution_layer(256, 3)
        self.vgg_maxpool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv4 = self._make_convolution_layer(512, 3)
        self.vgg_maxpool4 = nn.MaxPool2d((2,2), stride=(2,2))

        # Deconv / UNet
        self.unet_upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv1 = self._make_convolution_layer(256, 3)
        self.unet_upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv2 = self._make_convolution_layer(128, 2)
        self.unet_upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv3 = self._make_convolution_layer(64, 2)

        self.last_conv = nn.Conv2d(9, 3, kernel_size=(1,1), padding=1)
        self.last_sigmoid = nn.Sigmoid()


    def forward(self, x):
        # VGG16
        x1 = self.vgg_conv1(x)
        x = self.vgg_maxpool1(x1)
        x2 = self.vgg_conv2(x)
        x = self.vgg_maxpool2(x2)
        x3 = self.vgg_conv3(x)
        x = self.vgg_maxpool3(x3)
        x = self.vgg_conv4(x)
        x = self.vgg_maxpool4(x)

        # Deconv / UNet
        x = torch.concat([self.unet_upsample1(x), x3], dim=1)
        x = self.unet_conv1(x)
        x = torch.concat([self.unet_upsample2(x), x2], dim=1)
        x = self.unet_conv2(x)
        x = torch.concat([self.unet_upsample3(x), x3], dim=1)
        x = self.unet_conv3(x)

        x = self.last_conv(x)
        x = self.last_sigmoid(x)

        return x


if __name__ == '__main__':
    tinymodel = TrackNet()
