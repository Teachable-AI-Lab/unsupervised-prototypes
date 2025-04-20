from encoder_decoder.resnet_encoder import Encoder, BasicBlockEnc
from encoder_decoder.resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
import torch
import torch.nn as nn


####################################
#       MAE Encoder                # For images size >= 64x64
####################################
# TODO

####################################
#       ResNet-50 Encoder          # For images size >= 64x64
####################################
# TODO

####################################
#       ResNet-18 Encoder          # For images size >= 64x64
####################################
class ResNet18Encoder(nn.Module): 
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])

    def forward(self, x):
        return self.encoder(x)
    
####################################
#       ResNet-34 Encoder          # For images size >= 64x64
####################################
class ResNet34Encoder(nn.Module): 
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3])

    def forward(self, x):
        return self.encoder(x)
    
####################################
#       ResNet-18 Encoder          # For images size < 64x64
####################################
class ResNet18LightEncoder(nn.Module): 
    def __init__(self):
        super(ResNet18LightEncoder, self).__init__()
        self.encoder = LightEncoder(LightBasicBlockEnc, [2, 2, 2]) 

    def forward(self, x):
        return self.encoder(x)
    
####################################
#       ResNet-20 Encoder          # For images size < 64x64
####################################
class ResNet20LightEncoder(nn.Module): 
    def __init__(self):
        super(ResNet20LightEncoder, self).__init__()
        self.encoder = LightEncoder(LightBasicBlockEnc, [3, 3, 3]) 

    def forward(self, x):
        return self.encoder(x)
    
##########################
#       MLP Encoder      #
##########################
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

#######################################
#         28*28 Encoder         #
#######################################
class Encoder28x28(nn.Module):
    def __init__(self):
        super(Encoder28x28, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # output: 8x14x14
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # output: 16x7x7
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),  # output: 32x3x3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
#######################################
#          Omniglot Encoder         #
#######################################
class OmniglotEncoder(nn.Module):
    def __init__(self):
        super(OmniglotEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 28→28
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 28→13  ⟵ switched to 4×4
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 13→13
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 13→5   ⟵ switched to 4×4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 5→5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 5→1    ⟵ switched to 4×4
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    

# ####################################
#           VGG Encoder            #
# ####################################
class VGGEncoder(nn.Module): # only for cifar10
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # output: 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # output: 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # downsample: 64x16x16
        )
        # Block 2: from 16x16 to 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # downsample: 128x8x8
        )
        # Block 3: from 8x8 to 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # downsample: 256x4x4
        )
        # Block 4: from 4x4 to 2x2
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # downsample: 512x2x2
        )
        # Block 5: from 2x2 to 1x1 without average pooling
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=2),  # kernel_size=2 reduces 2x2 --> 1x1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)  # [B, 64, 16, 16]
        x = self.block2(x)  # [B, 128, 8, 8]
        x = self.block3(x)  # [B, 256, 4, 4]
        x = self.block4(x)  # [B, 512, 2, 2]
        x = self.block5(x)  # [B, 64, 1, 1]
        return x