from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        image = np.transpose(sample['image'], (2, 0, 1))  # [3, H, W]
        sample['image'] = torch.from_numpy(image) / 255.
        bg_mask = np.transpose(sample['bg_mask'], (2, 0, 1))  # [1, H, W]
        sample['bg_mask'] = torch.from_numpy(bg_mask)
        
        if "origin_size" in sample.keys():
            sample['origin_size'] = torch.from_numpy(np.array(sample['origin_size']))
        if 'fg_mask' in sample.keys():
            bg_mask = np.transpose(sample['fg_mask'], (2, 0, 1))  # [1, H, W]
            sample['fg_mask'] = torch.from_numpy(bg_mask)
        if 'fg_image' in sample.keys():
            fg_image = np.transpose(sample['fg_image'], (2, 0, 1))  # [3, H, W]
            sample['fg_image'] = torch.from_numpy(fg_image) / 255.
            
            
        return sample

class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['image'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['image'] = np.lib.pad(sample['image'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['bg_mask'] = np.lib.pad(sample['bg_mask'],
                                         ((top_pad, 0), (0, right_pad), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'fg_mask' in sample.keys():
                sample['fg_mask'] = np.lib.pad(sample['fg_mask'],
                                ((top_pad, 0), (0, right_pad), (0, 0)),
                                mode='constant',
                                constant_values=0)
            if 'fg_image' in sample.keys():
                sample['fg_image'] = np.lib.pad(sample['fg_image'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['image'] = self.crop_img(sample['image'])
            sample['bg_mask'] = self.crop_img(sample['bg_mask'])
            if 'fg_mask' in sample.keys():
                sample['fg_mask'] = self.crop_img(sample['fg_mask'])
            if 'fg_image' in sample.keys():
                sample['fg_image'] = self.crop_img(sample['fg_image'])



        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['image'] = np.copy(np.flipud(sample['image']))
            if 'fg_image' in sample.keys():
                sample['fg_image'] = np.copy(np.flipud(sample['fg_image']))

        return sample


class ToPILImage(object):
    def __call__(self, sample):
        sample['image'] = Image.fromarray(sample['image'].astype('uint8'))
        if 'fg_image' in sample.keys():
            sample['fg_image'] = Image.fromarray(sample['fg_image'].astype('uint8'))
        return sample


class ToNumpyArray(object):
    def __call__(self, sample):
        sample['image'] = np.array(sample['image']).astype(np.float32)
        if 'fg_image' in sample.keys():
            sample['fg_image'] = np.array(sample['fg_image']).astype(np.float32)
        return sample

# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['image'] = F.adjust_contrast(sample['image'], contrast_factor)

            if 'fg_image' in sample.keys():
                sample['fg_image'] = F.adjust_contrast(sample['fg_image'], contrast_factor)           

        return sample


class RandomGamma(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)  # adopted from FlowNet
            sample['image'] = F.adjust_gamma(sample['image'], gamma)

            if 'fg_image' in sample.keys():
                sample['fg_image'] = F.adjust_gamma(sample['fg_image'], gamma) 

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            sample['image'] = F.adjust_brightness(sample['image'], brightness)

            if 'fg_image' in sample.keys():
                sample['fg_image'] = F.adjust_brightness(sample['fg_image'], brightness)             

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            sample['image'] = F.adjust_hue(sample['image'], hue)

            if 'fg_image' in sample.keys():
                sample['fg_image'] = F.adjust_hue(sample['fg_image'], hue) 

        return sample


class RandomSaturation(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['image'] = F.adjust_saturation(sample['image'], saturation)

            if 'fg_image' in sample.keys():
                sample['fg_image'] = F.adjust_saturation(sample['fg_image'], saturation)
        
        return sample


class RandomColor(object):
    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample