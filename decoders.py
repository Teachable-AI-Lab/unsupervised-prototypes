from encoder_decoder.resnet_decoder import Decoder, BasicBlockDec
from encoder_decoder.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec
import torch
import torch.nn as nn

####################################
#       MAE Decoder                # For images size >= 64x64
####################################
# TODO

####################################
#       ResNet-50 Decoder       # For images size >= 64x64
####################################
# TODO

####################################
#       ResNet-18 Decoder          # For images size >= 64x64
####################################
class ResNet18Decoder(nn.Module): 
    def __init__(self, latent_dim, c, h, w):
        super(ResNet18Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.c = c
        self.h = h
        self.w = w
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.c * self.h * self.w),
            nn.ReLU(),
            nn.Unflatten(1, (self.c, self.h, self.w)),
            Decoder(BasicBlockDec, [2, 2, 2, 2]),
        )

    def forward(self, x):
        return self.decoder(x)

####################################
#       ResNet-34 Decoder          # For images size >= 64x64
####################################
class ResNet34Decoder(nn.Module): 
    def __init__(self, latent_dim, c, h, w):
        super(ResNet34Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.c = c
        self.h = h
        self.w = w
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.c * self.h * self.w),
            nn.ReLU(),
            nn.Unflatten(1, (self.c, self.h, self.w)),
            Decoder(BasicBlockDec, [3, 4, 6, 3])
        )

    def forward(self, x):
        return self.decoder(x)
    
####################################
#       ResNet-18 Decoder          # For images size < 64x64
####################################
class ResNet18LightDecoder(nn.Module): 
    def __init__(self):
        super(ResNet18LightDecoder, self).__init__()
        self.decoder = LightDecoder(LightBasicBlockDec, [2, 2, 2]) 

    def forward(self, x):
        return self.decoder(x)
    
####################################
#       ResNet-20 Decoder          # For images size < 64x64
####################################
class ResNet20LightDecoder(nn.Module): 
    def __init__(self, latent_dim, c, h, w):
        super(ResNet20LightDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.c = c
        self.h = h
        self.w = w
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.c * self.h * self.w),
            nn.ReLU(),
            nn.Unflatten(1, (self.c, self.h, self.w)),
            LightDecoder(LightBasicBlockDec, [3, 3, 3]) 
        )
    def forward(self, x):
        return self.decoder(x)
    
####################################
#       MLP Decoder                #
####################################
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.decoder(x)
    
#######################################
#         28*28 Decoder         #
#######################################
class Decoder28x28(nn.Module):
    def __init__(self):
        super(Decoder28x28, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=0),  # output: 16x7x7
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 8x14x14
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 1x28x28
            nn.ReLU(inplace=True)  # final activation for pixel values
        )
    def forward(self, x):
        x = self.decoder(x)
        return x


##########################################
#          VGG Decoder                #
##########################################

class VGGDecoder(nn.Module):
    def __init__(self):
        super(VGGDecoder, self).__init__()
        self.block5d = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=2),  # 64x1x1 -> 512x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Inverse of Block 4: from 2x2 to 4x4
        self.block4d = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),  # 512x2x2 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # adjust channels: 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Inverse of Block 3: from 4x4 to 8x8
        self.block3d = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 256x4x4 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # adjust channels: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Inverse of Block 2: from 8x8 to 16x16
        self.block2d = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # 128x8x8 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # adjust channels: 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Inverse of Block 1: from 16x16 to 32x32
        self.block1d = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 64x16x16 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # final conv to get 3-channel output 
            nn.ReLU(inplace=True)  # use ReLU if image pixels are scaled [-1,1]
        )

    def forward(self, x):
        x = self.block5d(x)  # [B, 512, 2, 2]
        x = self.block4d(x)  # [B, 256, 4, 4]
        x = self.block3d(x)  # [B, 128, 8, 8]
        x = self.block2d(x)  # [B, 64, 16, 16]
        x = self.block1d(x)  # [B, 3, 32, 32]
        return x