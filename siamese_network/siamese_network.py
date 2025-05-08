import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, crop_fusion_module, scene_comparison_module, encode_obj_with_JPM=False):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.crop_fusion_mlp = crop_fusion_module
        self.scene_comparison_mlp = scene_comparison_module
        self.encode_obj_with_JPM = encode_obj_with_JPM

    def forward(self, crop1, crop2, crop3, scene):
        """
        crop1, crop2, crop3: (B, 3, H, W)
        scene: (B, 3, H, W)
        Returns: (B, 1) — binary output between 0 and 1
        """

        # embed all crops and scene with shared encoder
        if self.encode_obj_with_JPM:
            embed_1 = self.encoder(crop1, use_jpm=True)   # ViT: (B, 1152)
            embed_2 = self.encoder(crop2, use_jpm=True)
            embed_3 = self.encoder(crop3, use_jpm=True)
        else:
            embed_1 = self.encoder(crop1)   # ResNet18: (B, 512)
            embed_2 = self.encoder(crop2)
            embed_3 = self.encoder(crop3)

        scene_embed = self.encoder(scene)  # ResNet18: (B, 512) ViT: (B, 384)

        # fuse all crops with MLP
        fused_crop = self.crop_fusion_mlp(embed_1, embed_2, embed_3)  # (B, 512)

        # predict if the object is in the scene based on embeddings with MLP
        output = self.scene_comparison_mlp(fused_crop, scene_embed)  # (B, 1)

        return output
