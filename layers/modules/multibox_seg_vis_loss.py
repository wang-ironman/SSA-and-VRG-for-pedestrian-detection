# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v
from distutils.version import LooseVersion
from ..box_utils import match,match_vis,log_sum_exp
import numpy as np
np.set_printoptions(threshold=np.inf)


class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes, size, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos,neg_pos_l, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        # self.negpos_ratio_l = neg_pos_l
        self.neg_overlap = neg_overlap
        cfg = v["640_480_base512"]
        self.variance = cfg['variance']

    def forward(self, predictions,predictions_vis, targets, targets_vis, seg_targets, seg_visible_targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors, fcn_output = predictions
        loc_data_vis, conf_data_vis, priors, fcn_visible_output = predictions_vis

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        loc_t_vis = torch.Tensor(num, num_priors, 4)
        conf_t_vis = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            truths_vis = targets_vis[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            labels_vis = targets_vis[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
            match_vis(self.threshold, truths_vis, defaults, self.variance, labels_vis,
                  loc_t_vis, conf_t_vis, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            loc_t_vis = loc_t_vis.cuda()
            conf_t_vis = conf_t_vis.cuda()
        # wrap bbox_targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)  # 匹配后正例为1，负例为0
        loc_t_vis = Variable(loc_t_vis, requires_grad=False)
        conf_t_vis = Variable(conf_t_vis, requires_grad=False)

        pos_all = conf_t > 0
        pos_vis = conf_t_vis > 0

        pos = pos_all & pos_vis
        mask = pos.gt(0)
        pos_mask = pos[mask]

        if (pos_mask.size()[0] != 0):
            pos_all = pos
            pos_vis = pos




        pos_idx = pos_all.unsqueeze(pos_all.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p,loc_t, size_average=False)

        pos_idx_vis = pos_vis.unsqueeze(pos_vis.dim()).expand_as(loc_data_vis)
        loc_p_vis = loc_data_vis[pos_idx_vis].view(-1, 4)
        loc_t_vis = loc_t_vis[pos_idx_vis].view(-1, 4)
        loss_l_vis = F.smooth_l1_loss(loc_p_vis, loc_t_vis, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos_all] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos_all.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos_all.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)


        pos_idx = pos_all.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        pos_idx_vis = pos_vis.unsqueeze(2).expand_as(conf_data_vis)
        neg_idx_vis = neg.unsqueeze(2).expand_as(conf_data_vis)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)

        bbox_targets_weighted = conf_t[(pos_all + neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, bbox_targets_weighted, size_average=False)

        conf_p_vis = conf_data_vis[(pos_idx_vis + neg_idx_vis).gt(0)].view(-1, self.num_classes)
        bbox_targets_weighted_vis = conf_t_vis[(pos_vis + neg).gt(0)]

        loss_c_vis = F.cross_entropy(conf_p_vis, bbox_targets_weighted_vis, size_average=False)

        loss_seg = cross_entropy2d(fcn_output, seg_targets)
        loss_seg_visible = cross_entropy2d(fcn_visible_output, seg_visible_targets)

        N = num_pos.data.sum()
        N = N.float()

        loss_l = loss_l / N
        loss_c = loss_c / N
        loss_l_vis = loss_l_vis / N
        loss_c_vis = loss_c_vis / N
        return loss_l, loss_c,loss_l_vis, loss_c_vis, loss_seg, loss_seg_visible

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)

    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0

    target = target[mask]

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        N = mask.data.sum()
        N = N.float()
        loss = loss / N

    return loss


class FSFocalLoss(nn.Module):
    def __init__(self, configer):
        super(FSFocalLoss, self).__init__()
        self.configer = configer

    def forward(self, output, target, **kwargs):
        self.y = self.configer.get('focal_loss', 'y')
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss
