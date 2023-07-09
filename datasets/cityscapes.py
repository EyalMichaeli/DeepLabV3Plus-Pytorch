import json
import os
from collections import namedtuple
from pathlib import Path
import random
import warnings

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import logging


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    # if u want to get the class name given the train_id, use this: classes[train_id].name
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    train_id_to_class_name = {c.train_id: c.name for c in classes}
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, aug_json=None, 
                 aug_sample_ratio: float = None, train_sample_ratio: float = 1.0):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.is_train = split == 'train'
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))


        # # add cities from the test set of cityscapes, to see if the model will improve from the extra data
        # if split == 'train':
        #     logging.info("Adding extra data from the test set of cityscapes")
        #     test_cities_list = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']
        #     for test_city in test_cities_list:
        #         extra_img_dir = self.root + '/leftImg8bit/test/' + test_city + '/'
        #         extra_target_dir = self.root + '/gtFine/test/' + test_city + '/'
        #         for file_name in os.listdir(extra_img_dir):
        #             self.images.append(os.path.join(extra_img_dir, file_name))
        #             target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
        #                                             self._get_target_suffix(self.mode, self.target_type))
        #             self.targets.append(os.path.join(extra_target_dir, target_name))
            
        #     # print one example:
        #     logging.info(self.images[0])
        #     logging.info(self.targets[0])
        #     logging.info(self.images[2975])
        #     logging.info(self.targets[2975])


        logging.info(f"Found {len(self.images)} {split} images")
        logging.info(f"Found {len(self.targets)} {split} targets")


        # use only a subset of the images for training, if train_sample_ratio < 1
        if split == 'train' and train_sample_ratio < 1:
            assert train_sample_ratio > 0, "train_sample_ratio must be > 0"
            subset_size = int(len(self.images) * train_sample_ratio)
            self.images = self.images[:subset_size]
            self.targets = self.targets[:subset_size]
            logging.info(f"With ratio {train_sample_ratio}, using only {subset_size} images for training, out of {len(self.images)}")

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


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image_path = self.images[index]
        if self.is_train:
            if self.aug_json:
                ratio_used_aug = 0
                if random.random() < self.aug_sample_ratio:
                    original_image_path = image_path
                    aug_img_files = self.aug_json.get(Path(image_path).name, [image_path])  # if image_path is not in aug_json, returns image_path
                    aug_img_files = [image_path] if len(aug_img_files) == 0 else aug_img_files
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

        image = Image.open(image_path).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)