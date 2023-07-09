import json
import os
from pathlib import Path
import random
import sys
import tarfile
import collections
import warnings
import torch.utils.data as data
import shutil
import numpy as np
import logging

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}




def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    classes = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'dining table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted plant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'TV monitor',
        255: 'background'
    }
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None, train_sample_ratio: float = 1.0, 
                 aug_json=None, aug_sample_ratio: float = None):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'
        
        self.root = os.path.expanduser(root)
        self.train_sample_ratio = train_sample_ratio
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.is_train = image_set == 'train'
        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join( self.root, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        
        # use only a subset of the images for training, if train_sample_ratio < 1
        if image_set == 'train' and train_sample_ratio < 1:
            assert train_sample_ratio > 0, "train_sample_ratio must be > 0"
            subset_size = int(len(self.images) * train_sample_ratio)
            logging.info(f"With ratio {train_sample_ratio}, using only {subset_size} images for training, out of {len(self.images)}")
            self.images = self.images[:subset_size]
            self.masks = self.masks[:subset_size]

        if self.is_train and aug_json:
            assert aug_sample_ratio is not None
            assert aug_sample_ratio > 0 and aug_sample_ratio <= 1
            with open(aug_json, 'r') as f:
                self.aug_json = json.load(f)
            # leave only keys that thier values (which is a list) is not empty
            self.aug_json = {k: v for k, v in self.aug_json.items() if v}

            self.aug_sample_ratio = aug_sample_ratio
            self.times_used_orig_images = 0
            self.times_used_aug_images = 0

            logging.info(f"Using augmented images with ratio {aug_sample_ratio}")
            logging.info(f"There are {len(self.aug_json)} augmented images, out of {len(self.images)} original images, \n which is {round(len(self.aug_json)/len(self.images), 2)*100}% of the original images")
            logging.info(f"json file: {aug_json}")

        else:
            self.aug_json = None
            logging.info('Not using augmented images')


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        image_path = self.images[index]
        if self.is_train:
            if self.aug_json:
                ratio_used_aug = 0
                if random.random() < self.aug_sample_ratio:
                    original_image_path = image_path
                    aug_img_files = self.aug_json.get(Path(image_path).name, [image_path])  # if image_path is not in aug_json, returns image_path
                    aug_img_files = [image_path] if len(aug_img_files) == 0 else aug_img_files  # if image_path key in the json returns an enpty list, use current image_path
                    image_path = random.choice(aug_img_files)
                    if original_image_path == image_path:  # didn't use augmented image
                        #print("Augmented image not found in aug_json")
                        self.times_used_orig_images += 1

                    else:  # used augmented image
                        #print(f"Using Augmented image found in aug_json: {image_path}")
                        self.times_used_aug_images += 1
                    pass

                else:
                    self.times_used_orig_images += 1

                ratio_used_aug = self.times_used_aug_images / (self.times_used_orig_images + self.times_used_aug_images)

                if index % 100 == 0 and ratio_used_aug < self.aug_sample_ratio / 3:  # check every 100 iters. e.g, if aug_sample_ratio = 0.3, then ratio_used_aug should not be less than 0.1
                    warn = f"Using augmented images is probably lacking, ratio: {ratio_used_aug:.4f} when it should be around {self.aug_sample_ratio}"
                    warnings.warn(warn)
                    logging.info(f"self.times_used_aug_images = {self.times_used_aug_images}, self.times_used_orig_images = {self.times_used_orig_images}")
                    
                # every 500 iters, print the ratio of original images to augmented images
                if index % 1000 == 0:
                    logging.info(f"Used augmented images {(ratio_used_aug*100):.4f}% of the time")

        # print(f"image_path: {image_path}")
        img = Image.open(image_path).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)