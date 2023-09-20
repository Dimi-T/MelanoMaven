import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):                                                 # custom model for binary classification based on VGG16 architecture
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.vgg16(weights = None).features.children())[:-1]) # remove last layer
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 4096),                                        
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)                                       # flatten
        x = self.classifier(x)
        return x