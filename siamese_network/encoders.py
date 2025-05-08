import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import timm


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


class ViTEncoder(nn.Module):
    def __init__(self, dropout=0.1, freeze=False):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.embed_dim = self.vit.embed_dim  # 384
        self.dropout = nn.Dropout(dropout)
        # remove classification head, just CLS
        self.vit.head = nn.Identity()

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x, use_jpm=False):
        """
        x: (B, 3, H, W)
        Returns:
          - (B, 384) if not using JPM
          - (B, 384*3) if using JPM
        """
        if not use_jpm:
            # standard forward pass: extract CLS token
            tokens = self.vit.forward_features(x)  # (B, 197, 384)
            cls = tokens[:, 0]                    # (B, 384)
            return self.dropout(cls)

        else:
            # JPM forward pass: shuffle patches 3 times, extract CLS each time
            cls_tokens = []
            for _ in range(3):
                tokens = self._forward_jigsaw(x)  # (B, 197, 384)
                cls = tokens[:, 0]                # (B, 384)
                cls_tokens.append(cls)

            concat_cls = torch.cat(cls_tokens, dim=1) # (B, 1152)
            return self.dropout(concat_cls)

    def _forward_jigsaw(self, x):
        """
        Randomly shuffles the patch tokens (excluding CLS) and returns features
        """
        # forward up to patch + pos embed
        B = x.size(0)
        x = self.vit.patch_embed(x)            # (B, 196, 384)
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, 384)
        x = torch.cat((cls_token, x), dim=1)   # (B, 197, 384)
        x = x + self.vit.pos_embed             # (B, 197, 384)
        x = self.vit.pos_drop(x)

        # permute patch tokens
        patch_tokens = x[:, 1:]                # (B, 196, 384)
        idx = torch.randperm(patch_tokens.shape[1])
        patch_tokens = patch_tokens[:, idx]    # shuffle
        x = torch.cat([x[:, :1], patch_tokens], dim=1)  # re-attach CLS

        # pass through transformer
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x