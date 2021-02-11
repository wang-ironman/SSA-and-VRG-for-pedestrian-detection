# -*- coding: utf-8 -*-
import torch
from math import sqrt as sqrt
from itertools import product as product
import pdb

# 针对 640*480 这样宽高不一致的图片，进行代码编写

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()

        # print(cfg)
        # pdb.set_trace()
        self.image_size = cfg['min_dim']

        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        # version is v2_512 or v2_300
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        # pdb.set_trace()
        for k, f in enumerate(self.feature_maps):

            # 修改成 i j 分开的

            for i in range(f[0]):   # h
                for j in range(f[1]): # w
                    # 对应于各种不同的feature maps

                    f_k_w = self.image_size[0] / self.steps[k][1]   # 用到了steps ，但是没有做修正
                    f_k_h = self.image_size[1] / self.steps[k][0]


                    cx = (j + 0.5) / f_k_w
                    cy = (i + 0.5) / f_k_h

                    # rest of aspect ratios
                    for tmp_size in range(self.min_sizes[k], self.max_sizes[k], 10):
                        """
                        :type tmp_size from min to max size
                        """
                        s_k_tmp_w = tmp_size / self.image_size[0]
                        s_k_tmp_h = tmp_size / self.image_size[1]

                        for ar in self.aspect_ratios[k]:
                            mean += [cx, cy, s_k_tmp_w* sqrt(ar), s_k_tmp_h / sqrt(ar)]


        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output
