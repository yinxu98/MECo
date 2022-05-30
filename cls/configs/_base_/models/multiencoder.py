# __vector_dim = 512  # res18,34
__vector_dim = 1024  # vgg
# __vector_dim = 2048  # res50

model = dict(
    type='MultiEncoder',
    # backbone=dict(
    #     type='MyResNetV1d',
    #     depth=34,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    # ),
    backbone=dict(
        type='MySSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
    ),
    predictor=dict(
        dim_in=__vector_dim,
        dim_mid=__vector_dim // 2,
        dim_out=__vector_dim,
    ),
)
