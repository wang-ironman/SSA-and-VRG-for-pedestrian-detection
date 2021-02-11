# -*- coding: utf-8 -*-
"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pdb
import PIL.Image
import pdb

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'person','background',)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class AnnotationTransform_vis(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target_vis, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res_vis = []
        for obj in target_vis.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res_vis += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res_vis  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class AnnotationTransform_caltech(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # print("voc0712 w,h")
            # print(width,height) # (640, 480)

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 target_vis_transform=None,dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.target_vis_transform = target_vis_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._annopath_vis = os.path.join('%s', 'Annotations_vis', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')  ## caltech jpg citypersons png
        self._seglbl = os.path.join('%s', 'SegmentationClass_weak', '%s.png')
        self._seglbl_visibile = os.path.join('%s', 'SegmentationClass_visible', '%s.png')
        self.ids = list()

        # for (year, name) in image_sets:
        #     rootpath = os.path.join(self.root, 'VOC' + year)
        #     for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + '0712')
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, gt_vis,seg,seg_vis, h, w = self.pull_item(index)
        # print("im type")
        # print(type(im))
        return im, gt, gt_vis, seg, seg_vis

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        #print('img_id',img_id)
        seg1 = PIL.Image.open(self._seglbl % img_id)
        seg = seg1.convert('P')
        seg1_vis = PIL.Image.open(self._seglbl_visibile % img_id)
        seg_vis = seg1_vis.convert('P')

        target = ET.parse(self._annopath % img_id).getroot()
        target_vis = ET.parse(self._annopath_vis % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.target_vis_transform is not None:
            target_vis = self.target_vis_transform(target_vis, width, height)
        if self.transform is not None:
            target = np.array(target)
            target_vis = np.array(target_vis)
            img, boxes, labels, boxes_vis, labels_vis,seg,seg_vis = self.transform(img, target[:, :4], target[:, 4],target_vis[:, :4], target_vis[:, 4],seg,seg_vis)

            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            target_vis = np.hstack((boxes_vis, np.expand_dims(labels_vis, axis=1)))

        seg = np.array(seg, dtype=np.int32)
        seg[seg == 1] = -1
        seg[seg == 2] = 1

        seg_vis = np.array(seg_vis, dtype=np.int32)
        seg_vis[seg_vis == 1] = -1
        seg_vis[seg_vis == 2] = 1

        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), target, target_vis,torch.from_numpy(seg).long(),torch.from_numpy(seg_vis).long(), height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        return img

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_anno_vis(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno_vis = ET.parse(self._annopath_vis % img_id).getroot()
        gt_vis = self.target_vis_transform(anno_vis, 1, 1)
        return img_id[1], gt_vis

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
    def pull_id(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        
        return img_id[1]


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    targets_vis = []
    imgs = []
    targets_seg = []
    targets_seg_vis = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        targets_vis.append(torch.FloatTensor(sample[2]))
        targets_seg.append(sample[3])
        targets_seg_vis.append(sample[4])
    return torch.stack(imgs, 0), targets, targets_vis, torch.stack(targets_seg, 0), torch.stack(targets_seg_vis, 0)

