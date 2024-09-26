from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import cv2



# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_mask(filename):

    mask = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    mask = mask>127
    mask = mask.astype(np.bool_)
    
    return mask

def resize_image(image,size,is_mask):
    original_shape = image.shape[:2]
    if not is_mask:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        resized_image = resized_image.astype(np.bool_)
    return resized_image, original_shape


def find_bounding_box(mask):
    # 找到前景的边界
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    
    return top, bottom, left, right


def get_resize_foreground_and_mask(image, mask):
    # 找到前景物体的边界
    bounding_box = find_bounding_box(mask)
    
    if bounding_box is None:
        # 如果mask全为0，返回原图和原Mask
        return image, mask
    
    top, bottom, left, right = bounding_box
    
    # 根据边界裁剪图像和mask
    cropped_image = image[top:bottom, left:right, :]
    cropped_mask = mask[top:bottom, left:right, :]
    
    # 将裁剪后的图像调整为原始大小
    cropped_image = (cropped_image).astype(np.uint8)
    resized_image = np.array(Image.fromarray(cropped_image).resize((image.shape[1], image.shape[0]), Image.BILINEAR))
    resized_image = resized_image.astype(np.float32)
    
    # 将裁剪后的Mask调整为原始大小
    resized_mask = np.array(Image.fromarray(cropped_mask.squeeze(-1)).resize((image.shape[1], image.shape[0]), Image.NEAREST))
    resized_mask = resized_mask[:, :, np.newaxis]  # 恢复Mask的第三个维度
    
    return resized_image, resized_mask