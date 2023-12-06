'''U-Net implementation in PyTorch.'''
import torch
import torch.nn as nn

from ..hparams import UNetHparams


class UNet(nn.Module):
    '''U-Net implementation in PyTorch.'''

    def __init__(self, hparams: UNetHparams):
        '''Initialize UNet.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            init_features (int): number of initial features
        '''
        super().__init__()

        self.hparams = hparams
        features = self.hparams.features
        self.encoder1 = UNet._block(self.hparams.in_channels, features, name='enc1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name='enc2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name='enc3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name='enc4')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name='bottleneck')

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name='dec4')
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name='dec3')
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name='dec2')
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name='dec1')

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=self.hparams.out_channels, kernel_size=1
        )

    @staticmethod
    def _block(in_channels, features, name):
        '''U-Net block.

        Args:
            in_channels (int): number of input channels
            features (int): number of output channels
            name (str): name of the block

        Returns:
            nn.Sequential: U-Net block
        '''
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        '''Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        '''
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    @torch.inference_mode()
    def inference(self, x):
        '''Inference.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        '''
        return self.forward(x)
