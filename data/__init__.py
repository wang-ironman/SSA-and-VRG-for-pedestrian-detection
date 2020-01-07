# -*- coding: utf-8 -*-
from .voc0712_seg import VOCDetection, AnnotationTransform,AnnotationTransform_vis, AnnotationTransform_caltech, detection_collate, VOC_CLASSES

# 弱语义分割标注
#from .voc_bbox_seg import VOCBboxSeg, AnnotationTransformAddSeg, detection_collate_add_seg, VOC_CLASSES_ADD_SEG


from .config import *
import cv2
import numpy as np


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

def base_transform_caltech(image, size, mean):
    if type(size) is int :
        x = cv2.resize(np.array(image), (size, size)).astype(np.float32)

    elif size[0] != 512 or size[0] != 300:
        x = cv2.resize(image, (size[0], size[1])).astype(np.float32)  # 参数输入是 宽×高×通道

    #print("x shape: ")
    #print(x.shape)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


class BaseTransformCaltech:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform_caltech(image, self.size, self.mean), boxes, labels
