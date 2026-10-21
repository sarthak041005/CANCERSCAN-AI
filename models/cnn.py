import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class CancerCNN(nn.Module):

    def __init__(self, num_classes, pretrained=True):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(weights=weights)

        n = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.50),
            nn.Linear(n, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    @property
    def feature_layer(self):
        return self.backbone.layer4[-1]