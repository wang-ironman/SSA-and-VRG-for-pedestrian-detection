import torch
from torch.autograd import Function
from ..box_utils import decode, nms,nms_v
from data import v


class Detect_v(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        cfg = v["640_480_base512"]
        self.variance = cfg['variance']
        self.output_vis = torch.zeros(1, self.num_classes, self.top_k, 5)

    def forward(self, loc_data_v, conf_data_v, prior_data,ids_all,count_all,conf_preds_all):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]


            fcn_output: seg result
        """
        # print('''batch_size {.1d}'''.format(loc_data.size(0)))
        num = loc_data_v.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output_vis.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data_v.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data_v.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            self.output_vis.expand_(num, self.num_classes, self.top_k, 5)
        # print('conf_p_v', conf_preds)
        # Decode predictions into bboxes.
        conf_preds = conf_preds_all
        for i in range(num):
            prior_data = prior_data.cuda()
            decoded_boxes = decode(loc_data_v[i], prior_data, self.variance)

            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            # print('conf_score:',conf_scores)
            num_det = 0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # print('score:',scores)
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                boxes = decoded_boxes[l_mask].view(-1, 4)
                print(boxes.size())
                # idx of highest scoring and non-overlapping boxes per class
                # ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # print(ids[:count].cpu().numpy())
                # scores = score_all
                ids = ids_all
                count = count_all.int()
                self.output_vis[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
                # print(self.output_vis.size())
        flt = self.output_vis.view(-1, 5)
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)


        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)

       
        return self.output_vis