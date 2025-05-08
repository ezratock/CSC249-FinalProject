import torch
import torch.nn as nn


class CropFusionMLP(nn.Module):
    def __init__(self, dropout=0.3, input_dim=512, fused_dim=512):
        super().__init__()

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 3, 1024),  # 1536 to 1024
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, fused_dim),     # 1024 to 512
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, crop1, crop2, crop3):
        x = torch.cat([crop1, crop2, crop3], dim=1)  # (B, 1536)
        fused = self.fusion_mlp(x)                   # (B, 512)
        return fused


class SceneComparisonMLP(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, obj_fusion, scene_embedding):
        x = torch.cat([obj_fusion, scene_embedding], dim=1)  # concatenate: (B, 1024)
        return self.classifier(x)  # output: (B, 1)

