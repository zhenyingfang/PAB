_base_ = [
    "../_base_/datasets/anet-1.3-multi/multi_i3d_label0.1_video0.1_point0.1.py",  # dataset config
]

model = dict(
    type="PointDETR",  # Done
    projection=dict(
        type="ConvSingleProj",
        in_channels=2048,
        out_channels=256,
        num_convs=1,
        conv_cfg=dict(kernel_size=1, padding=0),
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=None,
    ),
    transformer=dict(
        type="PointTadTRTransformer",
        num_proposals=100,
        num_classes=1,
        with_act_reg=True,
        roi_size=16,
        roi_extend_ratio=0.25,
        aux_loss=True,
        position_embedding=dict(
            type="PositionEmbeddingSine",
            num_pos_feats=256,
            temperature=10000,
            offset=-0.5,
            normalize=True,
        ),
        encoder=dict(
            type="DeformableDETREncoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            post_norm=False,
        ),
        decoder=dict(
            type="PointDeformableDETRDecoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            return_intermediate=True,
        ),
        loss=dict(
            type="PointSmoothL1Loss",
        ),
    ),
)


solver = dict(
    train=dict(batch_size=8, num_workers=4),
    train_video=dict(batch_size=8, num_workers=4),
    train_point=dict(batch_size=8, num_workers=4),
    val=dict(batch_size=8, num_workers=4),
    test=dict(batch_size=8, num_workers=4),
    clip_grad_norm=1,
)

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=10, max_epoch=30)

inference = dict(test_epoch=5, load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.35,
        max_seg_num=100,
        min_score=0.0001,
        multiclass=False,
        voting_thresh=0,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=5,
    val_eval_interval=5,
    val_start_epoch=10,
    end_epoch=30,
)

work_dir = "exps/omnitad/anet/i3d_label0.1_video0.1_point0.1"
