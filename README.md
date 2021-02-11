# ssd-for-pedestrian-detection

code for our paper"基于语义分割注意力和可见区域预测的行人检测网络".

Contributed by Lu Wang, Shuai Wang, Guofeng Zhang, Lisheng Xu.

## Getting Started

### installation
The code was tested on Ubuntu 16.04, with Python 3.6 and PyTorch v0.4.1.

1. Clone this repository.
  ~~~
  git clone https://github.com/wang-ironman/ssd-for-pedestrian-detection.git
  ~~~
2. Create virtual environment by conda.
  ~~~
  conda create -n pytorch0.4.1 python==3.6
  ~~~
2. Install pytorch0.4.1.
  ~~~
  conda install pytorch=0.4.1 cudatoolkit=9.0 torchvision -c pytorch
  ~~~
3. Install the requirements.
  ~~~
  pip install -r requirements.txt
  ~~~
  
## Training and Test

### Dataset Preparation

1. Download Caltech datasets. （[The download link of the pre-processed data is here](http://baidu.com)）Organize them in Dataset folder as follows:

    ~~~
    |-- data/
    |   |-- VOCdevkit/
    |       |-- VOC0712
    |           |-- Annotations
                |-- Annotations_vis
                |-- ImageSets
                |-- JPEGImages
                |-- SegmentationClass_visible
                |-- SegmentationClass_weak
    ~~~
2. Download the pre-trained model of VGG.
  ~~~
  [百度云链接](链接：https://pan.baidu.com/s/19nfPE-5FM743QmvY8_YW4g 提取码：xdcf)
  ~~~
  
### Training
An example traning as follow:

    ~~~
    cd $ROOT
    source activate pytorch0.4.1
    CUDA_VISIBLE_DEVICES=0 python train_vis_seg.py
    ~~~
### Test
An example test as follow:
    ~~~
    cd $ROOT
    source activate pytorch0.4.1
    CUDA_VISIBLE_DEVICES=0 python test_vis_seg_all.py --trained_model output/ssd640_0712_90000.pth
    ~~~
### Evalution
1. Pre-process the detection txt file for evaluation
    ```
    cd test_eval
    python stand.py
    python sort.py
    ```
2. Use matlab to evaluate the results.

