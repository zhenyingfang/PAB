import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import decoupled_cls_utils as utils


# copied from HR-Pro
class Reliable_Memory(nn.Module):
    def __init__(self, num_class, feat_dim):
        super(Reliable_Memory, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.proto_momentum = 0.001 
        self.proto_num = 1
        self.proto_vectors = torch.nn.Parameter(torch.randn([self.num_class, self.proto_num, self.feat_dim]), requires_grad=False)

    def update(self, feats, act_seq, vid_label):
        self.proto_vectors = self.proto_vectors.to(feats.device)
        feat_list = {}
        for b in range(act_seq.shape[0]):
            gt_class = torch.nonzero(vid_label[b]).cpu().squeeze(1).numpy()
            for c in gt_class:
                select_id = torch.nonzero(act_seq[b, :, c]).squeeze(1)
                if select_id.shape[0] > 0:
                    act_feat = feats[b, select_id, :]
                    if c not in feat_list.keys():
                        feat_list[c] = act_feat
                    else:
                        feat_list[c] = torch.cat((feat_list[c], act_feat))

        for c in feat_list.keys():
            if len(feat_list[c]) > 0:
                feat_update = feat_list[c].mean(dim=0, keepdim=True)
                self.proto_vectors[c] = (1 - self.proto_momentum) * self.proto_vectors[c] + self.proto_momentum * feat_update


# copied from HR-Pro
class Reliabilty_Aware_Block(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8, dim_feedforward=128):
        super(Reliabilty_Aware_Block, self).__init__()
        self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)

        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, attn_mask=None,):
        src = features.permute(2, 0, 1)
        q = k = src
        q = self.conv_query(features).permute(2, 0, 1)
        k = self.conv_key(features).permute(2, 0, 1)

        src2, attn = self.self_atten(q, k, src, attn_mask=attn_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.permute(1, 2, 0)
        return src, attn

  
class DecoupledClsEncoder(nn.Module):
    def __init__(self, feature_dim, drop_out=0.3, num_heads=8, dim_feedforward=128, layer_num=2):
        super(DecoupledClsEncoder, self).__init__()
        self.feature_dim = feature_dim

        RAB_args = {
            'drop_out': drop_out,
            'num_heads': num_heads,
            'dim_feedforward': dim_feedforward,
            'layer_num': layer_num
        }
        self.RAB = nn.ModuleList([
            Reliabilty_Aware_Block(
                input_dim=self.feature_dim,
                dropout=RAB_args['drop_out'],
                num_heads=RAB_args['num_heads'],
                dim_feedforward=RAB_args['dim_feedforward'])
            for i in range(RAB_args['layer_num'])
        ])

        self.feature_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, input_features, prototypes=None):
        '''
        input_feature: [B,T,F]
        prototypes: [C,1,F]
        '''
        B, T, F = input_features.shape
        input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
        prototypes = prototypes.to(input_features.device)                       #[C,1,F]
        prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
        if hasattr(self, 'RAB'):
            layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
            for layer in self.RAB:
                layer_features, _ = layer(layer_features)
            input_features = layer_features[:, :, :T]                           #[B,F,T]
        embeded_features = self.feature_embedding(input_features)               #[B,F,T]

        return embeded_features


class TextEncoder(nn.Module):
    def __init__(self, text_dim, feature_dim):
        super(TextEncoder, self).__init__()
        self.feature_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=feature_dim,
                out_channels=text_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features, text_proto):
        B, T, F = input_features.shape
        input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
        feature_embedding = self.feature_embedding(input_features)

        feature_embedding = feature_embedding.transpose(1, 2)
        text_proto = text_proto.transpose(0, 1).to(feature_embedding.dtype)

        text_logits = torch.matmul(feature_embedding, text_proto)
        text_logits = self.sigmoid(text_logits)

        return text_logits

class DecoupledCls(nn.Module):
    def __init__(self, feature_dim, drop_out=0.3, num_heads=8, dim_feedforward=256, layer_num=2, class_nums=200, r_act=8):
        super().__init__()
        self.r_act = r_act
        self.drop_out = drop_out
        self.class_nums = class_nums
        self.feature_dim = feature_dim
        self.memory = Reliable_Memory(self.class_nums, self.feature_dim)
        self.decoupled_encoder = DecoupledClsEncoder(
            feature_dim=self.feature_dim,
            drop_out=self.drop_out,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            layer_num=layer_num
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.class_nums+1, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        self.text_encoder = TextEncoder(
            text_dim=768,
            feature_dim=self.feature_dim
        )

        self.sigmoid = nn.Sigmoid()
        self.bce_criterion = nn.BCELoss(reduction='none')
    
    def forward(self, input_features, gt_decoupled_cls_labels=None, gt_segments=None, gt_labels=None, text_proto=None, **kwargs):
        # prepare targets
        vid_labels, point_labels, act_seed, bkg_seed = self._prepare_targets(
            feat=input_features,
            gt_decoupled_cls_labels=gt_decoupled_cls_labels,
            gt_segments=gt_segments
        )

        video_vid_labels, point_vid_labels, point_point_labels = None, None, None
        video_bs, point_bs = 0, 0
        if kwargs.get('data_video_dict', None) is not None:
            video_bs = kwargs['data_video_dict']['inputs'].shape[0]
            video_decls_labels = kwargs['data_video_dict']['decls_labels']
            video_vid_labels, _, _, _ = self._prepare_targets(
                feat=input_features,
                gt_decoupled_cls_labels=video_decls_labels,
            )
        if kwargs.get('data_point_dict', None) is not None:
            point_bs = kwargs['data_point_dict']['inputs'].shape[0]
            point_decls_labels = kwargs['data_point_dict']['decls_labels']
            point_point_labels = torch.stack(kwargs['data_point_dict']['point_labels']).float()
            point_vid_labels, _, _, _ = self._prepare_targets(
                feat=input_features,
                gt_decoupled_cls_labels=point_decls_labels,
            )
        cls_bs = input_features.shape[0] - video_bs - point_bs

        # >> Text Encoder and classifier
        text_logits = self.text_encoder(input_features, text_proto)
        
        # >> Encoder and classifier
        embeded_feature = self.decoupled_encoder(input_features, self.memory.proto_vectors)   #[B,F,T]
        cas = self.classifier(embeded_feature)                                      #[B,C+1,T]
        cas = cas.permute(0, 2, 1)                                                  #[B,T,C+1]
        cas = self.sigmoid(cas)                                                     #[B,T,C+1]
        # class-Specific activation sequence
        cas_S = cas[:, :, :-1]                                                      #[B,T,C]
        # class-Agnostic attention sequence (background)
        bkg_score = cas[:, :, -1]                                                   #[B,T]
        # >> Fusion
        cas_P = cas_S * (1 - bkg_score.unsqueeze(2))                                #[B,T,C]
        cas_fuse = torch.cat((cas_P, bkg_score.unsqueeze(2)), dim=2)                #[B,T,C+1]

        # >> Top-k pooling
        value, _ = cas_S.sort(descending=True, dim=1)
        k_act = max(1, input_features.shape[1] // self.r_act)
        topk_scores = value[:, :k_act, :]

        text_value, _ = text_logits.sort(descending=True, dim=1)
        text_topk_scores = text_value[:, :k_act, :]

        all_vid_labels = None
        if vid_labels is not None:
            all_vid_labels = vid_labels
        if video_vid_labels is not None:
            all_vid_labels = torch.cat((all_vid_labels, video_vid_labels), dim=0)
        if point_vid_labels is not None:
            all_vid_labels = torch.cat((all_vid_labels, point_vid_labels), dim=0)

        if all_vid_labels is None:
            vid_scores = torch.mean(topk_scores, dim=1)
            text_vid_scores = torch.mean(text_topk_scores, dim=1)
            vid_scores = 0.5 * vid_scores + 0.5 * text_vid_scores
        else:
            vid_scores = (torch.mean(topk_scores, dim=1) * all_vid_labels) + \
                        (torch.mean(cas_S, dim=1) * (1 - all_vid_labels))
            text_vid_scores = (torch.mean(text_topk_scores, dim=1) * all_vid_labels) + \
                        (torch.mean(text_logits, dim=1) * (1 - all_vid_labels))
        
        if vid_labels is not None:
            decoupled_cls_loss = self.cls_criterion(
                vid_score=vid_scores[:cls_bs],
                embeded_feature=embeded_feature[:cls_bs].permute(0, 2, 1),
                cas_fuse=cas_fuse[:cls_bs],
                vid_label=vid_labels,
                point_label=point_labels,
                act_seed=act_seed,
                bkg_seed=bkg_seed
            )

            video_loss_dict = self.cls_video_criterion(
                vid_score=text_vid_scores[:cls_bs],
                vid_label=vid_labels,
            )
            decoupled_cls_loss['loss_vid_text'] = video_loss_dict['loss_video_vid']

            if kwargs.get('data_video_dict', None) is not None:
                video_loss_dict = self.cls_video_criterion(
                    vid_score=vid_scores[cls_bs:cls_bs+video_bs],
                    vid_label=video_vid_labels,
                )
                decoupled_cls_loss.update(video_loss_dict)

                video_loss_dict = self.cls_video_criterion(
                    vid_score=text_vid_scores[cls_bs:cls_bs+video_bs],
                    vid_label=video_vid_labels,
                )
                decoupled_cls_loss['loss_vid_video_text'] = video_loss_dict['loss_video_vid']
            if kwargs.get('data_point_dict', None) is not None:
                point_loss_dict = self.cls_point_criterion(
                    vid_score=vid_scores[cls_bs+video_bs:],
                    embeded_feature=embeded_feature[cls_bs+video_bs:].permute(0, 2, 1),
                    cas_fuse=cas_fuse[cls_bs+video_bs:],
                    vid_label=point_vid_labels,
                    point_label=point_point_labels,
                )
                decoupled_cls_loss.update(point_loss_dict)

                video_loss_dict = self.cls_video_criterion(
                    vid_score=text_vid_scores[cls_bs+video_bs:],
                    vid_label=point_vid_labels,
                )
                decoupled_cls_loss['loss_vid_point_text'] = video_loss_dict['loss_video_vid']

            return vid_scores, decoupled_cls_loss
        else:
            ref_points, ref_point_masks, ref_cls_labels = get_pred_points(vid_scores, cas_fuse, gt_segments=gt_segments, gt_labels=gt_labels)
            ref_points, ref_point_masks = torch.tensor(ref_points).float(), torch.tensor(ref_point_masks).unsqueeze(2).float()
            ref_cls_labels = torch.tensor(ref_cls_labels).long().unsqueeze(2)
            return vid_scores, ref_points, ref_point_masks, ref_cls_labels


    def _prepare_targets(self, feat, gt_decoupled_cls_labels=None, gt_segments=None):
        vid_labels = None
        point_labels = None
        act_seed, bkg_seed = None, None
        if gt_decoupled_cls_labels is not None:
            now_device = gt_decoupled_cls_labels[0].device
            gt_nums = len(gt_decoupled_cls_labels)
            point_labels = torch.zeros((gt_nums, feat.shape[1], self.class_nums)).to(now_device)
            vid_labels = torch.zeros((gt_nums, self.class_nums)).to(now_device)
            for ggl in range(len(gt_decoupled_cls_labels)):
                gt_global_tmp = gt_decoupled_cls_labels[ggl].cpu().detach().numpy().tolist()
                for ggt in gt_global_tmp:
                    vid_labels[ggl][ggt] = 1.
            if gt_segments is not None:
                act_seed = torch.zeros((gt_nums, feat.shape[1], self.class_nums + 1)).to(now_device)
                bkg_seed = torch.zeros((gt_nums, feat.shape[1])).to(now_device)
                for ggl in range(len(gt_decoupled_cls_labels)):
                    gt_global_tmp = gt_decoupled_cls_labels[ggl].cpu().detach().numpy().tolist()
                    ggl_segments = gt_segments[ggl].cpu().detach().numpy()
                    for seg_idx in range(ggl_segments.shape[0]):
                        ggl_seg = ggl_segments[seg_idx]
                        # ggl_inter = int((ggl_seg[0] + ggl_seg[1]) / 2.)
                        ggl_start = max(0, int(ggl_seg[0]))
                        ggl_end = min(feat.shape[1], int(ggl_seg[1]))
                        ggl_len = ggl_end - ggl_start
                        ggl_inter = int(random.uniform(ggl_start, ggl_end))
                        point_labels[ggl][ggl_inter][gt_global_tmp[seg_idx]] = 1
                        for act_idx in range(ggl_start, ggl_end):
                            act_seed[ggl][act_idx][gt_global_tmp[seg_idx]] = 1
                    tmp_seed = act_seed[ggl].clone().sum(dim=-1)
                    tmp = torch.where(tmp_seed > 0, 0, 1)
                    tmp[0] = 1
                    tmp[-1] = 1
                    bkg_seed[ggl] = tmp
        
        return vid_labels, point_labels, act_seed, bkg_seed

    def cls_criterion(
            self,
            vid_score,
            embeded_feature,
            cas_fuse,
            vid_label,
            point_label,
            act_seed=None,
            bkg_seed=None
        ):
        point_label = torch.cat((point_label, torch.zeros((point_label.shape[0], point_label.shape[1], 1)).to(embeded_feature.device)), dim=2)
        if act_seed is None:
            act_seed, bkg_seed = utils.select_seed(cas_fuse[:, :, -1].detach().cpu(), point_label.detach().cpu())

        loss_dict = {}
        # >> base loss
        loss_vid, loss_frame, loss_frame_bkg = self.base_loss_func(act_seed, bkg_seed, vid_score, vid_label, cas_fuse, point_label)
        loss_dict["loss_vid"] = loss_vid * 0.5
        loss_dict["loss_frame"] = loss_frame * 0.5
        loss_dict["loss_frame_bkg"] = loss_frame_bkg * 1.0

        # >> feat loss
        loss_contrastive = self.feat_loss_func(embeded_feature, act_seed, bkg_seed, vid_label)
        loss_dict["loss_contrastive"] = loss_contrastive * 1.0

        # >> update memory
        self.memory.update(embeded_feature.detach(), act_seed, vid_label)

        # loss_total = 1 * loss_vid + 1 * loss_frame \
        #             + 1 * loss_frame_bkg + 1 * loss_contrastive
        # loss_dict["loss_total"] = loss_total

        return loss_dict

    def cls_video_criterion(
            self,
            vid_score,
            vid_label,
        ):
        loss_vid = self.base_loss_func(None, None, vid_score, vid_label, None, None)
        loss_dict = {}
        loss_dict["loss_video_vid"] = loss_vid * 10.0

        return loss_dict

    def cls_point_criterion(
            self,
            vid_score,
            embeded_feature,
            cas_fuse,
            vid_label,
            point_label,
        ):
        point_label = torch.cat((point_label, torch.zeros((point_label.shape[0], point_label.shape[1], 1)).to(embeded_feature.device)), dim=2)
        act_seed, bkg_seed = utils.select_seed(cas_fuse[:, :, -1].detach().cpu(), point_label.detach().cpu())

        loss_dict = {}
        # >> base loss
        loss_vid, loss_frame, loss_frame_bkg = self.base_loss_func(act_seed, bkg_seed, vid_score, vid_label, cas_fuse, point_label)
        loss_dict["loss_point_vid"] = loss_vid * 0.5
        loss_dict["loss_point_frame"] = loss_frame * 0.5
        loss_dict["loss_point_frame_bkg"] = loss_frame_bkg * 1.0

        # >> feat loss
        loss_contrastive = self.feat_loss_func(embeded_feature, act_seed, bkg_seed, vid_label)
        loss_dict["loss_point_contrastive"] = loss_contrastive * 1.0

        # >> update memory
        self.memory.update(embeded_feature.detach(), act_seed, vid_label)

        return loss_dict


    def base_loss_func(self, act_seed, bkg_seed, vid_score, vid_label, cas_sigmoid_fuse, point_anno):
        # >> video-level loss
        loss_vid = self.bce_criterion(vid_score, vid_label)
        loss_vid = loss_vid.mean()

        if act_seed is None:
            return loss_vid

        # >> frame-level loss
        loss_frame = torch.zeros(()).to(vid_label.device)
        loss_frame_bkg = torch.zeros(()).to(vid_label.device)
        # act frame loss
        act_seed = act_seed.to(vid_label.device)
        focal_weight_act = (1 - cas_sigmoid_fuse ) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)
        loss_frame = (((focal_weight_act * self.bce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act)
                        .sum(dim=2)).sum(dim=1) / (num_actions + 1e-6)).mean()
        # bkg frame loss
        bkg_seed = bkg_seed.unsqueeze(-1).to(vid_label.device)
        point_anno_bkg = torch.zeros_like(point_anno).to(vid_label.device)
        point_anno_bkg[:, :, -1] = 1
        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1).squeeze(1)
        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (((focal_weight_bkg * self.bce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg)
                            .sum(dim=2)).sum(dim=1) / (num_bkg + 1e-6)).mean()

        return loss_vid, loss_frame, loss_frame_bkg
    
    def feat_loss_func(self, embeded_feature, act_seed, bkg_seed, vid_label):
        loss_contra = torch.zeros(()).to(vid_label.device)
        proto_vectors = utils.norm(self.memory.proto_vectors.to(vid_label.device))                                        #[C,N,F]                                                             
        for b in range(act_seed.shape[0]):
            # >> extract pseudo-action/background features
            gt_class = torch.nonzero(vid_label[b]).squeeze(1)
            act_feat_lst = []
            for c in gt_class:
                act_feat_lst.append(utils.extract_region_feat(act_seed[b, :, c], embeded_feature[b, :, :]))
            bkg_feat = utils.extract_region_feat(bkg_seed[b].squeeze(-1), embeded_feature[b, :, :])
            if bkg_feat is None:
                continue
            
            # >> caculate similarity matrix
            if len(bkg_feat) == 0:
                continue
            bkg_feat = utils.norm(torch.cat(bkg_feat, 0))                                                            #[t_b,F]
            b_sim_matrix = torch.matmul(bkg_feat.unsqueeze(0).expand(self.class_nums, -1, -1), 
                                        torch.transpose(proto_vectors, 1, 2)) / 0.1                                  #[C,t_b,N]
            b_sim_matrix = torch.exp(b_sim_matrix).reshape(b_sim_matrix.shape[0], -1).mean(dim=-1)                   #[C]
            for idx, act_feat in enumerate(act_feat_lst):
                if act_feat is not None:
                    if len(act_feat) == 0:
                        continue
                    act_feat = utils.norm(torch.cat(act_feat, 0))                                                    #[t_a,F]
                    a_sim_matrix = torch.matmul(act_feat.unsqueeze(0).expand(self.class_nums, -1, -1), 
                                                torch.transpose(proto_vectors, 1, 2)) / 0.1                          #[C,t_a,N]
                    a_sim_matrix = torch.exp(a_sim_matrix).reshape(a_sim_matrix.shape[0], -1).mean(dim=-1)           #[C]                                                      

            # >> caculate contrastive loss
                    c = gt_class[idx]
                    loss_contra_act = - torch.log(a_sim_matrix[c] / a_sim_matrix.sum())
                    loss_contra_bkg = - torch.log(a_sim_matrix[c] / 
                                                 (a_sim_matrix[c] + b_sim_matrix[c]))
                    loss_contra += (0.5 * loss_contra_act + 0.5 * loss_contra_bkg)

            loss_contra = loss_contra / gt_class.shape[0]
        loss_contra = loss_contra / act_seed.shape[0]

        return loss_contra


def get_pred_points(vid_scores, cas_fuse, gt_segments=None, gt_labels=None):
    batch_points = []
    batch_masks = []
    batch_class_labels = []
    class_thresh = 0.7
    scale = 2.0
    num_class = 200
    act_thresh_cas = np.arange(0.0, 0.25, 0.025)
    act_thresh_agnostic = np.arange(0.5, 0.725, 0.025)

    _vid_score, _cas_fuse = vid_scores.detach(), cas_fuse.detach()
    for b in range(_cas_fuse.shape[0]):
        # >> caculate video-level prediction
        score_np = _vid_score[b].cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[np.where(score_np < class_thresh)] = 0
        pred_np[np.where(score_np >= class_thresh)] = 1
        if pred_np.sum() == 0:
            pred_np[np.argmax(score_np)] = 1

        # >> post-process
        cas_fuse = _cas_fuse[b]
        num_segments = _cas_fuse[b].shape[0]
        max_len = num_segments
        # class-specific score
        cas_S = cas_fuse[:, :-1]
        pred = np.where(score_np >= class_thresh)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])
        if len(pred) > 5:
            pred = np.argsort(score_np)[-5:]

        if gt_labels is not None:
            tmp_pred_np = gt_labels[b].cpu().detach().numpy().tolist()
            tmp_pred_np = list(set(tmp_pred_np))
            tmp_pred_np = np.array(tmp_pred_np)
            pred = tmp_pred_np

        cas_pred = cas_S.cpu().numpy()[:, pred]   
        cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
        cas_pred = utils.upgrade_resolution(cas_pred, scale)
        # class-agnostic score
        agnostic_score = 1 - cas_fuse[:, -1].unsqueeze(1)
        agnostic_score = agnostic_score.expand((-1, num_class))
        agnostic_score = agnostic_score.cpu().numpy()[:, pred]
        agnostic_score = np.reshape(agnostic_score, (num_segments, -1, 1))
        agnostic_score = utils.upgrade_resolution(agnostic_score, scale)
        
        # >> generate proposals
        proposal_dict = {}
        for i in range(len(act_thresh_cas)):
            cas_temp = cas_pred.copy()
            zero_location = np.where(cas_temp[:, :, 0] < act_thresh_cas[i])
            cas_temp[zero_location] = 0

            seg_list = []
            for c in range(len(pred)):
                pos = np.where(cas_temp[:, c, 0] > 0)
                seg_list.append(pos)
            proposals = utils.get_proposal_oic_seg(seg_list, cas_temp, score_np, pred)
            for i in range(len(proposals)):
                class_id = proposals[i][0][2]
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                proposal_dict[class_id] += proposals[i]

        for i in range(len(act_thresh_agnostic)):
            cas_temp = cas_pred.copy()
            agnostic_score_temp = agnostic_score.copy()
            zero_location = np.where(agnostic_score_temp[:, :, 0] < act_thresh_agnostic[i])
            agnostic_score_temp[zero_location] = 0

            seg_list = []
            for c in range(len(pred)):
                pos = np.where(agnostic_score_temp[:, c, 0] > 0)
                seg_list.append(pos)
            proposals = utils.get_proposal_oic_seg(seg_list, cas_temp, score_np, pred)
            for i in range(len(proposals)):
                class_id = proposals[i][0][2]
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                proposal_dict[class_id] += proposals[i]

        new_proposal_dict = dict()
        for pd_key, pd_value in proposal_dict.items():
            new_proposal_dict[pd_key] = utils.soft_nms(pd_value, sigma=0.3)
        
        key_len = len(new_proposal_dict.keys())
        npd_keys = list(new_proposal_dict.keys())
        item_points, item_mask, item_class_label = [], [], []

        total_samples = 100
        base_samples = total_samples // key_len
        extra_samples = total_samples % key_len

        sampled_values = []
        for i in range(key_len):
            num_samples = base_samples + (1 if i < extra_samples else 0)
            tmp_points, tmp_mask, tmp_class_label = proposal_sample_points(new_proposal_dict[npd_keys[i]][:num_samples], num_proposals=num_samples, max_len=max_len, repeat_num=1, class_label=npd_keys[i])
            if i == 0:
                item_points = tmp_points
                item_mask = tmp_mask
                item_class_label = tmp_class_label
            else:
                item_points.extend(tmp_points)
                item_mask.extend(tmp_mask)
                item_class_label.extend(tmp_class_label)

        batch_points.append(item_points)
        batch_masks.append(item_mask)
        batch_class_labels.append(item_class_label)
    
    return batch_points, batch_masks, batch_class_labels


def proposal_sample_points(pps, num_proposals=150, max_len=192, repeat_num=3, class_label=None):
    random_points = []
    point_mask = np.zeros((num_proposals, ))
    if class_label is not None:
        point_class_labels = [class_label for _ in range(num_proposals)]

    for i, pp in enumerate(pps):
        selected_list = np.linspace(pp[0], pp[1], repeat_num + 2)[1:-1]
        for j, fg_point in enumerate(selected_list):
            random_points.append([fg_point])
            point_mask[i * repeat_num + j] = 1

    while len(random_points) < num_proposals:
        bg_point = random.uniform(0, max_len)
        random_points.append([bg_point])

    return random_points, point_mask.tolist(), point_class_labels
