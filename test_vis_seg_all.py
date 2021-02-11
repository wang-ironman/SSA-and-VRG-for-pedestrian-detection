# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, VOC_CLASSES as labelmap
from PIL import Image
from data import AnnotationTransform_vis, AnnotationTransform_caltech, VOCDetection, detection_collate,BaseTransformCaltech, VOCroot, VOC_CLASSES
import torch.utils.data as data
from ssd_seg_vis_best import build_ssd
from log import log
import time
import pdb
import numpy as np
print(VOCroot)
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', required=True,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to test model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def set_seed(seed):
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(47)
cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    # pdb.set_trace()
    time_tup = time.localtime(time.time())
    format_time = '%Y_%m_%d_%a_%H_%M_%S'
    cur_time = time.strftime(format_time, time_tup)

    filename = save_folder + cur_time + '.txt'
    num_images = len(testset)
    for i in range(num_images):
        
        log.l.info('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id = testset.pull_id(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0), volatile=True)


        if cuda:
            x = x.cuda()

        y, y_vis = net(x)      # forward pass

        detections = y.data
        detections_vis = y_vis.data
        # scale each detection back up to the image   要将检测结果转换回原图上
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0

        for i in range(detections.size(1)):
            j = 0
            while (detections[0, i, j, 0] > 0) & (detections_vis[0, i, j, 0] > 0):

                score = detections[0, i, j, 0]
                score_vis = detections_vis[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pt_vis = (detections_vis[0, i, j, 1:] * scale).cpu().numpy()
                coords_vis = (pt_vis[0], pt_vis[1], pt_vis[2], pt_vis[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    score = score.cpu().numpy()
                    score_vis = score_vis.cpu().numpy()
                    f.write(img_id + ' ' + str(np.round(score,4)))
                    for c in coords:
                        f.write(' ')
                        f.write(str(np.round(c,2)))
                    f.write(' ')
                    f.write('0')
                    f.write('\n')

                    f.write(img_id + ' ' + str(np.round(score_vis, 4)))
                    for c_v in coords_vis:
                        f.write(' ')
                        f.write(str(np.round(c_v, 2)))
                    f.write(' ')
                    f.write('1')
                    f.write('\n')
                    pass
                j += 1
            print(j)


if __name__ == '__main__':

    # load net
    num_classes = 2  # +1 background
    net = build_ssd('test', [640,480], num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    log.l.info('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('0712', 'test')], None, AnnotationTransform_caltech())
    if args.cuda:
        net = net.cuda()

    # evaluation

    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransformCaltech(net.size, (106.6, 110.3, 107.7)),
             thresh=args.visual_threshold)
