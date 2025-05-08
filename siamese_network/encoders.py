import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Encoder(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()

        # pretrained ImageNet weights
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # remove the final ImageNet classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 512, 1, 1)
        x = self.flatten(x)            # (B, 512)
        return x
