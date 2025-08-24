import random
import numpy as np
import torch
import torch.nn as nn
from .layers import MLP, inverse_sigmoid
from ..builder import TRANSFORMERS
from .point_deformable_transformer import PointDeformableDETRTransformer
from ..utils.bbox_tools import proposal_se_to_cw
from ..roi_heads.roi_extractors.align1d.align import Align1DLayer
from .decoupled_layes.decoupled_cls import DecoupledCls


@TRANSFORMERS.register_module()
class PointTadTRTransformer(PointDeformableDETRTransformer):
    def __init__(
        self,
        num_proposals,
        num_classes,
        position_embedding=None,
        encoder=None,
        decoder=None,
        aux_loss=True,
        loss=None,
        with_act_reg=True,
        roi_size=16,
        roi_extend_ratio=0.25,
    ):
        super(PointTadTRTransformer, self).__init__(
            two_stage_num_proposals=num_proposals,
            num_classes=num_classes,
            position_embedding=position_embedding,
            encoder=encoder,
            decoder=decoder,
            aux_loss=aux_loss,
            loss=loss,
            with_box_refine=True,
            as_two_stage=False,
        )

        self.label_enc = nn.Embedding(200, self.encoder.embed_dim)
        self.num_proposals = num_proposals
        self.decoupled_cls = DecoupledCls(
            feature_dim=self.encoder.embed_dim
        )

        self.with_act_reg = with_act_reg
        if self.with_act_reg:  # RoI alignment
            hidden_dim = self.encoder.embed_dim
            self.roi_extend_ratio = roi_extend_ratio
            self.roi_extractor = Align1DLayer(roi_size)
            self.actionness_pred = nn.Sequential(
                nn.Linear(roi_size * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _to_roi_align_format(rois, T, roi_extend_ratio=1):
        """Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        """
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * (roi_extend_ratio * 2 + 1)
        rois_abs = torch.cat((rois_center - rois_size / 2, rois_center + rois_size / 2), dim=2) * T
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind.float(), rois_abs), dim=2)
        # NOTE: stop gradient here to stabilize training
        return rois_abs.view((B * N, 3)).detach()

    def forward_train(self, x, masks, gt_segments=None, gt_labels=None, decls_labels=None, is_training=True, metas=None, **kwargs):
        # The input of TadTR's transformer is single scale feature
        # x: [bs, c, t], masks: [bs, t], padding is 1.

        detr_invaild_len = 0
        detr_valid_len = masks.shape[0]
        if kwargs.get('data_video_dict', None) is not None:
            detr_invaild_len += kwargs['data_video_dict']['inputs'].shape[0]
        if kwargs.get('data_point_dict', None) is not None:
            detr_invaild_len += kwargs['data_point_dict']['inputs'].shape[0]
        
        if detr_invaild_len > 0:
            detr_valid_len = detr_valid_len - detr_invaild_len
            masks = masks[:detr_valid_len]

        # Here we set masks to be all False
        masks = torch.zeros_like(masks, dtype=torch.bool)
        max_len = masks.shape[-1]

        feat = x.permute(0, 2, 1)  # [bs, c, t] -> [bs, t, c]

        # >> decoupled cls
        decoupled_scores = None
        if 'text_proto' in kwargs.keys():
            text_proto = kwargs['text_proto'][0]
            del kwargs['text_proto']
        else:
            text_proto = metas[0]['text_proto']
        if is_training:
            decoupled_scores, decoupled_loss = self.decoupled_cls(
                input_features=feat,
                gt_decoupled_cls_labels=decls_labels,
                gt_segments=gt_segments,
                text_proto=text_proto,
                **kwargs
            )

            feat = feat[:detr_valid_len]

            # get random points
            ref_points, ref_point_masks, new_gt_segments, ref_cls_labels = self.generate_point(gt_segments, masks.shape[-1], x.device, repeat_num=10, gt_labels=gt_labels)
            # ref_points = ref_points.round()
            # >>> get psedu ref_points
            _, pse_ref_points, pse_ref_point_masks, pse_ref_cls_labels = self.decoupled_cls(
                input_features=feat.detach(),
                gt_segments=gt_segments,
                text_proto=text_proto,
                # gt_labels=decls_labels
            )
            if random.random() < 0.5:
                ref_points, ref_point_masks, new_gt_segments, ref_cls_labels = self.ge_pse_ref_point(gt_segments, pse_ref_points, gt_labels, pse_ref_cls_labels)
        else:
            gt_labels = []
            if metas[0].get('gt_video_labels', None) is not None:
                for meta in metas:
                    gt_labels.append(meta['gt_video_labels'])
            if metas[0].get('gt_points', None) is not None:
                for meta in metas:
                    gt_labels.append(meta['gt_labels'])
            if len(gt_labels) < 1:
                gt_labels = None
            decoupled_scores, ref_points, ref_point_masks, pse_ref_cls_labels = self.decoupled_cls(
                input_features=feat,
                # gt_segments=gt_segments,
                gt_labels=gt_labels,
                text_proto=text_proto,
            )

            ref_points, ref_point_masks, ref_cls_labels = ref_points.to(x.device), ref_point_masks.to(x.device), pse_ref_cls_labels.to(x.device)

        ref_points = torch.round(ref_points * 10) / 10.0
        ref_points = ref_points.float()

        pos_embed = self.position_embedding(masks) + self.level_embeds[0].view(1, 1, -1)  # [bs, t, c]

        lengths = torch.as_tensor([feat.shape[1]], dtype=torch.long, device=feat.device)
        level_start_index = lengths.new_zeros((1,))

        valid_ratios = self.get_valid_ratio(masks)[:, None]  # [bs, 1]
        reference_points = self.get_reference_points(lengths, valid_ratios, device=feat.device)  # [bs, t, 1]

        memory = self.encoder(
            query=feat,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks,
            spatial_shapes=lengths,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape

        tgt_embed = self.tgt_embed.weight  # nq, embed
        target = tgt_embed
        target = target.unsqueeze(0).expand(bs, -1, -1)

        proto_vectors = self.decoupled_cls.memory.proto_vectors
        # label_embed = self.label_enc(ref_cls_labels.squeeze(2))
        label_embed = proto_vectors.squeeze(1)[ref_cls_labels.squeeze(2)]
        label_embed = label_embed.detach()
        if target.shape[1] < label_embed.shape[1]:
            target = target[:, :label_embed.shape[1], :]
        # target = label_embed
        target = target + label_embed

        reference_points = ref_points / max_len

        if not is_training:
            target = target[:, :ref_points.shape[1], :]
        query = target
        query_pos = None

        # decoder
        inter_states, inter_references_out = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=masks,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 1
            spatial_shapes=lengths,  # nlvl
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,
            max_len=max_len,
            **kwargs,
        )

        #  Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            reference = reference_points if lvl == 0 else inter_references_out[lvl - 1]
            # reference = inverse_sigmoid(reference)
            reference = reference * max_len
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 2:
                tmp += reference
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference.squeeze(-1)
            # outputs_coord = tmp.sigmoid()
            outputs_coord = tmp / max_len
            # outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)  # tensor shape: [num_decoder_layers, bs, num_query, num_class]
        outputs_coord = torch.stack(outputs_coords)  # tensor shape: [num_decoder_layers, bs, num_query, 2]

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        output['cls_labels'] = ref_cls_labels

        if decoupled_scores is not None:
            output['pred_decls'] = decoupled_scores

        if self.with_act_reg:
            # perform RoIAlign
            B, N = outputs_coord[-1].shape[:2]
            rois = self._to_roi_align_format(outputs_coord[-1], memory.shape[1], roi_extend_ratio=self.roi_extend_ratio)
            roi_features = self.roi_extractor(memory.permute(0, 2, 1), rois).view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)
            output["pred_actionness"] = pred_actionness

        if is_training:
            if self.aux_loss:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            detr_loss = self.losses(output, masks, ref_point_masks, gt_segments, gt_labels, new_gt_segments)
            detr_loss.update(decoupled_loss)
            return detr_loss
        else:
            return output

    def losses(self, outputs, masks, ref_point_masks, gt_segments, gt_labels, new_gt_segments):
        new_gt_segments, gt_labels = self.prepare_targets(masks, new_gt_segments, gt_labels)
        loss_dict = self.criterion(outputs, gt_segments, gt_labels, ref_point_masks, new_gt_segments)
        return loss_dict

    @torch.no_grad()
    def prepare_targets(self, masks, gt_segments, gt_labels):
        gt_segments = [proposal_se_to_cw(bboxes / masks.shape[-1]) for bboxes in gt_segments]  # normalize gt_segments
        gt_labels = [labels.long() for labels in gt_labels]
        return gt_segments, gt_labels

    def forward_test(self, x, masks, gt_segments=None, gt_labels=None, **kwargs):
        return self.forward_train(x, masks, is_training=False, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)

    def generate_point(self, gt_segments, max_len, device, repeat_num=5, shuffle=True, gt_labels=None):
        ref_point_list = []
        point_masks = []
        new_gt_segments = []
        ref_cls_labels = []

        repeat_num = repeat_num

        for gt_item, gt_label in zip(gt_segments, gt_labels):
            point_mask = torch.zeros((self.num_proposals, 1))
            tmp_gt = torch.zeros((self.num_proposals, 2))

            if shuffle:
                if gt_item.size(0) > (self.num_proposals / repeat_num):
                    indices = list(range(gt_item.size(0)))
                    random.shuffle(indices)
                    gt_item = gt_item[indices]
                    gt_label = gt_label[indices]
                    gt_item = gt_item[:self.num_proposals // repeat_num, ...]
                    gt_label = gt_label[:self.num_proposals // repeat_num, ...]

            N = min(self.num_proposals, gt_item.size(0))

            random_points = []
            intervals = []
            cls_labels = []
            for i in range(N):
                start, end = gt_item[i]
                intervals.append((start.item(), end.item()))

                for j in range(repeat_num):
                    fg_point = random.uniform(start.item(), end.item())
                    random_points.append([fg_point])
                    cls_labels.append(gt_label[i].item())
                    point_mask[i * repeat_num + j][0] = 1
                    tmp_gt[i * repeat_num + j] = gt_item[i].clone()

            # intervals.sort()
            merged_intervals = []
            for start, end in intervals:
                merged_intervals.append((start, end))

            max_time = 0
            while len(random_points) < self.num_proposals:
                bg_point = random.uniform(0, max_len)
                valid = True
                for start, end in merged_intervals:
                    if start <= bg_point <= end:
                        valid = False
                        break
                if valid:
                    random_points.append([bg_point])
                    # cls_labels.append(random.randint(0, 199))
                    cls_labels.append(gt_label[random.randint(0, N-1)].item())
                max_time += 1
                if max_time > 1000:
                    break
            if len(random_points) < self.num_proposals:
                while len(random_points) < self.num_proposals:
                    bg_point = random.uniform(0, max_len)

                    valid = False
                    valid_idx = -1
                    for start, end in merged_intervals:
                        valid_idx += 1
                        if start <= bg_point <= end:
                            valid = True
                            break
                    
                    now_len = len(random_points)
                    random_points.append([bg_point])
                    
                    if valid:
                        point_mask[now_len][0] = 1
                        tmp_gt[now_len] = gt_item[valid_idx].clone()
                        cls_labels.append(gt_label[valid_idx].item())
                    else:
                        cls_labels.append(random.randint(0, 199))

            tmp_gt = tmp_gt.to(device)
            if shuffle:
                random_points = random_points[:self.num_proposals]
                cls_labels = cls_labels[:self.num_proposals]
                indices = list(range(len(random_points)))
                random.shuffle(indices)
                random_points = [random_points[i] for i in indices]
                cls_labels = [cls_labels[i] for i in indices]
                point_mask = point_mask[indices]
                tmp_gt = tmp_gt[indices]

            ref_point_list.append(random_points[:self.num_proposals])
            ref_cls_labels.append(cls_labels[:self.num_proposals])
            point_masks.append(point_mask)
            new_gt_segments.append(tmp_gt)

        ref_point_masks = torch.stack(point_masks).to(device)
        ref_points = torch.tensor(ref_point_list).to(device).float()
        ref_cls_labels = torch.tensor(ref_cls_labels).to(device).long().unsqueeze(2)
        
        return ref_points, ref_point_masks, new_gt_segments, ref_cls_labels

    def ge_pse_ref_point(self, gt_segments, pse_ref_points, gt_labels, pse_ref_cls_labels):
        pse_ref_points = pse_ref_points.to(gt_segments[0].device)
        pse_ref_masks = torch.zeros_like(pse_ref_points).to(gt_segments[0].device)
        pse_gt_segments = []
        for i in range(len(gt_segments)):
            gt_segment = gt_segments[i]
            pse_ref_point = pse_ref_points[i]
            pse_gt_segment = torch.zeros(pse_ref_point.size(0), 2).to(gt_segment.device)
            for j in range(pse_ref_point.size(0)):
                now_point = pse_ref_point[j][0].item()
                for k in range(gt_segment.size(0)):
                    gt_seg = gt_segment[k]
                    if now_point >= gt_seg[0].item() and now_point <= gt_seg[1].item():
                        pse_ref_masks[i][j][0] = 1
                        pse_gt_segment[j] = gt_seg
                        pse_ref_cls_labels[i][j] = gt_labels[i][k]
                        break
            pse_gt_segments.append(pse_gt_segment)
        
        return pse_ref_points, pse_ref_masks, pse_gt_segments, pse_ref_cls_labels.to(gt_segments[0].device)
