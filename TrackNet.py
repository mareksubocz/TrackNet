import torch
from torch import nn


class TrackNet(nn.Module):
    def _make_convolution_sublayer(self, in_channels, out_channels, dropout_rate=0.0):
        layer = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        ]
        if dropout_rate > 1e-15:
            layer.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layer)


    def _make_convolution_layer(self, in_channels, out_channels, num, dropout_rate=0.0):
        layers = []
        layers.append(self._make_convolution_sublayer(in_channels, out_channels, dropout_rate=dropout_rate))
        for _ in range(num-1):
            layers.append(self._make_convolution_sublayer(out_channels, out_channels, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)


    def __init__(self, dropout_rate = 0.0):
        super().__init__()

        # VGG16
        self.vgg_conv1 = self._make_convolution_layer(9, 64, 2, dropout_rate=dropout_rate)
        self.vgg_maxpool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv2 = self._make_convolution_layer(64, 128, 2, dropout_rate=dropout_rate)
        self.vgg_maxpool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv3 = self._make_convolution_layer(128, 256, 3, dropout_rate=dropout_rate)
        self.vgg_maxpool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv4 = self._make_convolution_layer(256, 512, 3, dropout_rate=dropout_rate)

        # Deconv / UNet
        self.unet_upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv1 = self._make_convolution_layer(768, 256, 3, dropout_rate=dropout_rate)
        self.unet_upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv2 = self._make_convolution_layer(384, 128, 2, dropout_rate=dropout_rate)
        self.unet_upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv3 = self._make_convolution_layer(192, 64, 2, dropout_rate=dropout_rate)

        #FIXME: probably should have 3 channels, not 9
        self.last_conv = nn.Conv2d(64, 3, kernel_size=(1,1), padding="same")
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

        # Deconv / UNet
        x = torch.concat([self.unet_upsample1(x), x3], dim=1)
        x = self.unet_conv1(x)
        x = torch.concat([self.unet_upsample2(x), x2], dim=1)
        x = self.unet_conv2(x)
        x = torch.concat([self.unet_upsample3(x), x1], dim=1)
        x = self.unet_conv3(x)

        x = self.last_conv(x)
        x = self.last_sigmoid(x)

        return x


    def save(self, path, whole_model=False):
        if whole_model:
            torch.save(self, path)
        else:
            torch.save(self.state_dict(), path)


    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))


if __name__ == '__main__':
    model = TrackNet()
