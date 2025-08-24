#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/point_tadtr_multi/i3d_label0.1.py

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/gen_pse_label.py \
    configs/point_tadtr_multi/i3d_label0.1.py \
    --checkpoint exps/omnitad/anet/i3d_label0.1/gpu1_id0/checkpoint/epoch_24.pth

python tools/generate_pse_label.py \
    --semi_anno_path data/ablation_annos/new_i3d_0.1.json \
    --pse_anno_path exps/omnitad/anet/i3d_label0.1/gpu1_id0/result_detection.json \
    --output_file data/ablation_annos/i3d_pse_label0.1.json

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py \
    configs/actionformer/anet_i3d_pse_multi_0.1.py

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
#     tools/train.py \
#     configs/point_tadtr_multi/i3d_label0.1_video0.1_point0.1.py

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
#     tools/gen_pse_label.py \
#     configs/point_tadtr_multi/i3d_label0.1_video0.1_point0.1.py \
#     --checkpoint exps/omnitad/anet/i3d_label0.1_video0.1_point0.1/gpu1_id0/checkpoint/epoch_24.pth

# python tools/generate_pse_label.py \
#     --semi_anno_path data/ablation_annos/new_i3d_0.1.json \
#     --pse_anno_path exps/omnitad/anet/i3d_label0.1_video0.1_point0.1/gpu1_id0/result_detection.json \
#     --output_file data/ablation_annos/i3d_pse_label0.1_video0.1_point0.1.json

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
#     tools/train.py \
#     configs/actionformer/anet_i3d_pse_multi_label0.1_video0.1_point0.1.py
