import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, crop_fusion_module, scene_comparison_module):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.crop_fusion_mlp = crop_fusion_module
        self.scene_comparison_mlp = scene_comparison_module

    def forward(self, crop1, crop2, crop3, scene):
        """
        crop1, crop2, crop3: (B, 3, H, W)
        scene: (B, 3, H, W)
        Returns: (B, 1) — binary output between 0 and 1
        """

        # embed all crops and scene with shared ResNet18 encoder
        embed_1 = self.encoder(crop1)   # (B, 512)
        embed_2 = self.encoder(crop2)   # (B, 512)
        embed_3 = self.encoder(crop3)   # (B, 512)
        scene_embed = self.encoder(scene)  # (B, 512)

        # fuse all crops with MLP
        fused_crop = self.crop_fusion_mlp(embed_1, embed_2, embed_3)  # (B, 512)

        # predict if the object is in the scene based on embeddings with MLP
        output = self.scene_comparison_mlp(fused_crop, scene_embed)  # (B, 1)

        return output
