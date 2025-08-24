import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.iou_tools import compute_giou_torch, compute_iou_torch
from ..builder import LOSSES
from .focal_loss import FocalLoss


@LOSSES.register_module()
class PointSmoothL1Loss(nn.Module):
    def __init__(
        self,
        losses = ["class", "boxes"],
    ):
        super().__init__()
        self.losses = losses
        self.focal_loss = FocalLoss()
    
    def get_loss(self, loss, outputs, gt_segments, gt_labels, masks, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, gt_segments, gt_labels, masks, **kwargs)

    def loss_labels(self, outputs, gt_segments, gt_labels, masks):
        inputs = outputs['pred_logits']
        targets = masks
        loss_class = self.focal_loss(inputs, targets, reduction='mean')
        loss_class = loss_class * 2.0
        losses = {'loss_class': loss_class}
        return losses

    def loss_boxes(self, outputs, gt_segments, gt_labels, masks):
        inputs = outputs['pred_boxes']
        masks = masks.squeeze(-1).long().bool()
        inputs = inputs[masks]
        # targets = torch.cat(gt_segments).repeat_interleave(5, dim=0)
        targets = torch.stack(gt_segments).to(inputs.device)[masks]
        loss = F.l1_loss(inputs, targets, reduction='none')
        l1_loss = loss.mean() * 5.0
        losses = {'loss_box': l1_loss}

        loss_giou = 1 - torch.diag(compute_giou_torch(proposal_cw_to_se(targets), proposal_cw_to_se(inputs)))
        losses["loss_giou"] = loss_giou.mean() * 2.0

        return losses

    def loss_actionness(self, outputs, gt_segments, gt_labels, masks):
        assert "pred_actionness" in outputs

        # inputs = outputs['pred_boxes']
        # masks = masks.squeeze(-1).long().bool()
        # inputs = inputs[masks]
        # targets = torch.cat(gt_segments).repeat_interleave(5, dim=0)

        # Compute GT IoU
        gt_iou = []
        for gt_segment, pred_boxes in zip(gt_segments, outputs['pred_boxes']):
            iou = compute_iou_torch(proposal_cw_to_se(gt_segment), proposal_cw_to_se(pred_boxes))
            gt_iou.append(iou.max(dim=1)[0])
        gt_iou = torch.cat(gt_iou, dim=0)  # [bs*num_queries]

        pred_iou = outputs["pred_actionness"].view(-1)  # [bs*num_queries]
        loss_actionness = F.l1_loss(pred_iou, gt_iou.detach())

        losses = {}
        losses["loss_actionness"] = loss_actionness * 4.0
        return losses

    def forward(
        self,
        outputs,
        gt_segments,
        gt_labels,
        masks,
        new_gt_segments
    ):
        losses = {}
        for loss in self.losses:
            loss_tmp = self.get_loss(loss, outputs, new_gt_segments, gt_labels, masks)
            losses.update(loss_tmp)
        
        actionness_loss = self.loss_actionness(outputs, new_gt_segments, gt_labels, masks)
        losses.update(actionness_loss)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, new_gt_segments, gt_labels, masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
