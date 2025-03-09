# define any encoders and decoders here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import 

##################################################
# ENCODERS
##################################################

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, representation_dim=512):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the fully connected layer and pooling to extract features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Optional projection layer to adjust the dimension of the representation
        self.fc = nn.Linear(512, representation_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # Flatten to [batch_size, 512]
        x = self.fc(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, pretrained=False, representation_dim=2048):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the fully connected layer and pooling to extract features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Optional projection layer to adjust the dimension of the representation
        self.fc = nn.Linear(2048, representation_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # Flatten to [batch_size, 2048]
        x = self.fc(x)
        return x
    
# class 