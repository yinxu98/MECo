import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_backbone


class MultiEncoder(nn.Module):
    def __init__(self, cfg):
        super(MultiEncoder, self).__init__()
        self.backbone_type = cfg.backbone.type
        self.backbone_depth = cfg.backbone.depth
        self.backbone = self._init_backbone(cfg.backbone)
        self.backbone_ = self._init_backbone(cfg.backbone)
        self.predictor = self._init_predictor(cfg.predictor)

    def _init_backbone(self, cfg):
        return build_backbone(cfg)

    def _init_predictor(self, cfg):
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
            r1 = self.backbone(x)[-1]
            r2 = self.backbone_(x)[-1]

        elif self.backbone_type == 'MySSDVGG':
            if self.backbone_depth == 11:
                r1 = self.backbone(x)
                r2 = self.backbone_(x)
            elif self.backbone_depth == 16:
                r1 = self.backbone(x)[-1]
                r2 = self.backbone_(x)[-1]

        r1 = self._global_pool(r1)
        r2 = self._global_pool(r2)

        p1 = self.predictor(r1)
        p2 = self.predictor(r2)

        return p1, p2, r1.detach(), r2.detach()


if __name__ == '__main__':
    from addict import Dict

    cfg = Dict(
        backbone=dict(
            type='MyResNetV1d',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
        ),
        predictor=dict(
            dim_in=512,
            dim_mid=256,
            dim_out=512,
        ),
    )
    model = MultiEncoder(cfg)
    data = torch.ones(size=[5, 3, 64, 64])
    output = model.backbone(data)
    for i in output:
        print(i.shape)
    output = model(data)
    for i in output:
        print(i.shape)
