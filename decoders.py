from encoder_decoder.resnet_decoder import Decoder, BasicBlockDec
from encoder_decoder.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(ResNet18Decoder, self).__init__()
        self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])


    def forward(self, x):
        return self.decoder(x)
    
####################################
#       Omniglot Decoder          #
###################################
class OmniglotDecoder(nn.Module):
    def __init__(self):
        super(OmniglotDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # 1→5    ⟵ 4×4 upsample
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 5→5
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 5→13   ⟵ 4×4 upsample
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 13→13
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 13→28  ⟵ 4×4 upsample (needs output_padding=1 to land on 28)
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 28→28
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

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
    def __init__(self, hidden_dim, input_dim):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim[0] * hidden_dim[1] * hidden_dim[2], 128),
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
    
####################################
#          RN Block                #
####################################
class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()

        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(in_channels=fin, out_channels=self.fhidden, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=self.fhidden, out_channels=self.fout, kernel_size=3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels=fin, out_channels=self.fout, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.fin)
        self.bn1 = nn.BatchNorm2d(self.fhidden)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(F.relu(self.bn0(x)))
        dx = self.conv_1(F.relu(self.bn1(dx)))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
    
class Resnet_Decoder(nn.Module):
    def __init__(self, s0=2, nf=8, nf_max=256, size=32):
        super(Resnet_Decoder, self).__init__()

        self.s0 = s0
        self.nf = nf  
        self.nf_max = nf_max 

        # Submodules
        nlayers = int(torch.log2(torch.tensor(size / s0).float()))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        # self.fc = nn.Linear(ndim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ]
        blocks += [
            ResnetBlock(nf, nf),
        ]
        self.resnet = nn.Sequential(*blocks)

        self.bn0 = nn.BatchNorm2d(nf)
        self.conv_img = nn.ConvTranspose2d(nf, 3, kernel_size=3, padding=1)


    def forward(self, z):
        # out = self.fc(z)
        # out = out.view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(z)
        out = self.conv_img(F.relu(self.bn0(out)))
        out = F.relu(out)
        return out
