import copy
import torch
import torch.nn as nn
from ...builder import TRANSFORMERS
from ..layers import BaseTransformerLayer, MultiheadAttention, MultiScaleDeformableAttention, FFN, inverse_sigmoid, get_sine_pos_embed, MLP


@TRANSFORMERS.register_module()
class PointDeformableDETRDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_points=4,
        attn_dropout=0.1,
        ffn_dim=2048,
        ffn_dropout=0.1,
        num_layers=6,
        num_feature_levels=4,
        batch_first=True,
        return_intermediate=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        self.num_layers = num_layers
        self.num_points = num_points
        self.num_feature_levels = num_feature_levels
        self.batch_first = batch_first
        self.return_intermediate = return_intermediate

        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(embed_dim // 2, embed_dim, embed_dim, 2)

        self._init_transformer_layers()
        self.bbox_embed = None
        self.class_embed = None

    def _init_transformer_layers(self):
        transformer_layers = BaseTransformerLayer(
            attn=[
                MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    attn_drop=self.attn_dropout,
                    batch_first=self.batch_first,
                ),
                MultiScaleDeformableAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.attn_dropout,
                    batch_first=self.batch_first,
                    num_points=self.num_points,
                    num_levels=self.num_feature_levels,
                ),
            ],
            ffn=FFN(
                embed_dim=self.embed_dim,
                ffn_dim=self.ffn_dim,
                ffn_drop=self.ffn_dropout,
            ),
            norm=nn.LayerNorm(
                normalized_shape=self.embed_dim,
            ),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )
        self.layers = nn.ModuleList([copy.deepcopy(transformer_layers) for _ in range(self.num_layers)])

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,  # nlvl
        level_start_index=None,  # nlvl
        valid_ratios=None,  # [bs, nlvl]
        max_len=192,
        **kwargs,
    ):
        output = query

        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 1

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None, :] * valid_ratios[:, None, :, None]
            # [B,T,lvl,2/1]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :1])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,  # nlvl
                level_start_index=level_start_index,  # nlvl
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + reference_points * max_len
                    new_reference_points = new_reference_points / max_len
                else:
                    assert reference_points.shape[-1] == 1
                    new_reference_points = tmp
                    new_reference_points[..., 0] = tmp[..., 0] + reference_points[..., 0] * max_len
                    new_reference_points = new_reference_points / max_len
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
