import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_backbone


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.backbone_type = cfg.backbone.type
        self.backbone_depth = cfg.backbone.depth
        self.backbone = self._init_backbone(cfg.backbone)
        self.classifier = self._init_classifier(cfg.classifier)

    def _init_backbone(self, cfg):
        backbone = build_backbone(cfg)
        if cfg.init_cfg.type == 'lin':
            for _, param in backbone.named_parameters():
                param.requires_grad = False
        return backbone

    def _init_classifier(self, cfg):
        dim_in = cfg.dim_in
        dim_mid = cfg.dim_mid
        dim_out = cfg.dim_out
        return nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.BatchNorm1d(dim_mid),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mid, dim_out),
        )

    def _global_pool(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(
            -1, num_channels)

    def forward(self, x):
        if self.backbone_type == 'MyResNetV1d':
            output = self.backbone(x)[-1]

        elif self.backbone_type == 'MySSDVGG':
            if self.backbone_depth == 11:
                output = self.backbone(x)
            elif self.backbone_depth == 16:
                output = self.backbone(x)[-1]

        output = self._global_pool(output)
        output = self.classifier(output)

        return output
