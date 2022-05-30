import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_backbone


class MultiEmbedding(nn.Module):
    def __init__(self, cfg):
        super(MultiEmbedding, self).__init__()
        self.ls_backbone = nn.ModuleList(
            self._init_backbone(cfg.backbone) for _ in range(cfg.n_backbone))
        self.ls_predictor = nn.ModuleList(
            self._init_predictor(cfg.predictor)
            for _ in range(cfg.n_predictor))

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
        ls_r = [backbone(x)[-1] for backbone in self.ls_backbone]

        ls_r = [self._global_pool(r) for r in ls_r]

        ls_p = [[predictor(r) for predictor in self.ls_predictor]
                for r in ls_r]

        ls_r = [r.detach() for r in ls_r]

        return ls_p, ls_r


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
            dim_in=2048,
            dim_mid=1024,
            dim_out=2048,
        ),
    )
    model = MultiEmbedding(cfg)
    data = torch.ones(size=[7, 3, 64, 64])
    output = model.backbone(data)
    for i in output:
        print(i.shape)
    output = model(data)
    for i in output:
        print(i.shape)
