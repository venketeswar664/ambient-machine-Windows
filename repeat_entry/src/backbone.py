import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Vendored copy of facebook/dinov2-base — repeat_entry/models/dinov2-base.
# Loading from a local directory removes any dependency on the HuggingFace
# hub cache location / network, which was breaking the nightly runner.
_LOCAL_DINOV2 = Path(__file__).resolve().parent.parent / "models" / "dinov2-base"


class DinoV2Backbone(nn.Module):

    def __init__(
        self,
        num_classes=172,
        model_name=None,
        unfreeze_n=2
    ):
        super().__init__()

        if model_name is None:
            model_name = (
                str(_LOCAL_DINOV2)
                if (_LOCAL_DINOV2 / "config.json").is_file()
                else "facebook/dinov2-base"
            )

        self.backbone = AutoModel.from_pretrained(model_name)

        embed_dim = self.backbone.config.hidden_size

        # ----------------------------------
        # Freeze backbone
        # ----------------------------------
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ----------------------------------
        # Unfreeze last N transformer blocks
        # ----------------------------------
        for layer in self.backbone.encoder.layer[-unfreeze_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # ----------------------------------
        # Embedding Head
        # ----------------------------------
        self.proj = nn.Sequential(

            nn.Linear(embed_dim, 512),

            nn.BatchNorm1d(512),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(512, 512)
        )

        # ----------------------------------
        # BNNeck
        # ----------------------------------
        self.bnneck = nn.BatchNorm1d(512)

        # standard ReID trick
        self.bnneck.bias.requires_grad_(False)

        # ----------------------------------
        # Classifier
        # ----------------------------------
        self.classifier = nn.Linear(
            512,
            num_classes,
            bias=False
        )

        print(
            f"✅ DINOv2 loaded | "
            f"last {unfreeze_n} layers unfrozen | "
            f"BNNeck enabled"
        )

    # ----------------------------------
    def forward(self, x):

        out = self.backbone(pixel_values=x)

        cls = out.last_hidden_state[:, 0]

        # ----------------------------------
        # Embedding branch
        # ----------------------------------
        feat = self.proj(cls)

        emb = F.normalize(
            feat,
            dim=1
        )

        # ----------------------------------
        # BNNeck only for classifier
        # ----------------------------------
        bn_feat = self.bnneck(feat)

        logits = self.classifier(bn_feat)

        return emb, logits

    # ----------------------------------
    def get_embedding(self, x):

        with torch.no_grad():

            out = self.backbone(pixel_values=x)

            cls = out.last_hidden_state[:, 0]

            feat = self.proj(cls)

            emb = F.normalize(
                feat,
                dim=1
            )

        return emb