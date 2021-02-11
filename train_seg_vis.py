# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from data import AnnotationTransform_vis, AnnotationTransform_caltech, VOCDetection, detection_collate, VOCroot, VOC_CLASSES

from utils.augmentations_vis_seg_all import SSDAugmentation
from layers.modules.multibox_seg_vis_loss import MultiBoxLoss
from ssd_seg_vis_best import build_ssd

#from IPython import embed
from log import log
import time
from utils.mkdir import mkdir
import pdb

print(torch.__version__)

def set_seed(seed):
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)

set_seed(47)
cudnn.deterministic = True
cudnn.benchmark = False

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=[640, 480], type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('-d', '--dataset', default='VOC',help='VOC or COCO dataset')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--data_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]

means = (106.6, 110.3, 107.7)
if args.dataset=='VOC':
    num_classes = 2
elif args.dataset=='kitti':
    num_classes = 2

stepvalues = (40000, 60000, 100000)
start_iter = 0

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', args.dim, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)

if args.resume:
    log.l.info('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
    start_iter = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])
else:
    resnet_weights = torch.load(args.save_folder + args.basenet)
    print(args.save_folder+args.basenet)
    log.l.info('Loading base network...')
    ssd_net.vgg.load_state_dict(resnet_weights)
    start_iter = 0

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# unicoe code
def weights_init_resnet(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight,1)
        nn.init.constant(m.bias,0)

if not args.resume:
    log.l.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    print("extras")
    print(ssd_net.extras)
    ssd_net.extras.apply(weights_init_resnet)
    ssd_net.loc.apply(weights_init_resnet)
    ssd_net.conf.apply(weights_init_resnet)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, args.dim, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def DatasetSync(dataset='VOC',split='training'):
    if dataset=='VOC':
        DataRoot=args.data_root
        dataset = VOCDetection(DataRoot, train_sets,
                               transform=SSDAugmentation(args.dim, means),
                               target_transform=AnnotationTransform_caltech(),
                               target_vis_transform=AnnotationTransform_vis())
    return dataset

def train():
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    loc_loss_vis = 0  # epoch
    conf_loss_vis = 0
    seg_loss = 0
    seg_visible_loss = 0
    epoch = 0
    log.l.info('Loading Dataset...')

    dataset=DatasetSync(dataset=args.dataset,split='training')

    epoch_size = len(dataset) / args.batch_size
    log.l.info('Training SSD on {}'.format(dataset.name))
    step_index = 0
    batch_iterator = None

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)

    lr = args.lr
    for iteration in range(start_iter, args.iterations + 1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            loc_loss_vis = 0
            conf_loss_vis = 0
            seg_loss = 0
            seg_visible_loss = 0
            epoch += 1

        # load train data
        images, targets, targets_vis, seg_targets, seg_visible_targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            targets_vis = [Variable(anno.cuda(), volatile=True) for anno in targets_vis]
            seg_targets = Variable(seg_targets.cuda())
            seg_visible_targets = Variable(seg_visible_targets.cuda())
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
            targets_vis = [Variable(anno, volatile=True) for anno in targets_vis]
            seg_targets = Variable(seg_targets)
            seg_visible_targets = Variable(seg_visible_targets)
        # forward
        t0 = time.time()
        out,out_vis = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c,loss_l_vis, loss_c_vis, loss_seg, loss_seg_visible = criterion(out,out_vis,targets,targets_vis, seg_targets, seg_visible_targets)
        alpha = 4
        loss_all = loss_l + loss_c
        loss_vis = loss_l_vis + loss_c_vis
        loss = loss_all + loss_vis + alpha * loss_seg + alpha * loss_seg_visible

        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        loc_loss_vis += loss_l_vis.item()
        conf_loss_vis += loss_c_vis.item()
        seg_loss += loss_seg.item()
        seg_visible_loss += loss_seg_visible.item()


        if iteration % 10 == 0:
            print(iteration,loss.item())
            log.l.info('''
                Timer: {:.3f} sec.\t LR: {}.\t Iter: {}.\t Loss: {:.4f}.\t Loss_a: {:.3f}.\t Loss_v: {:.3f}.\t Loss_seg:{:.3f}.\t Loss_seg_visible:{:.3f}.
                '''.format((t1-t0),lr,iteration,loss.item(),loss_all.item(),loss_vis.item(), alpha*loss_seg.item(), alpha*loss_seg_visible.item()))

        if iteration % 5000 == 0:
            log.l.info('Saving state, iter: {}'.format(iteration))
            mkdir("output4/")
            torch.save(ssd_net.state_dict(), 'output4/ssd640' + '_0712_' +
                       repr(iteration) + '.pth')

    torch.save(ssd_net.state_dict(), 'output4/ssd640' + '.pth')



def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https:/github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':

    train()
