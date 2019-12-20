""" Utils on generators / lists of ids to transform from strings to cropped images and masks """

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        im = resize_and_crop(Image.open(os.path.join(dir, id + suffix)), scale=scale)
        yield im


def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    mask_ids = []
    for i, id in enumerate(ids):
        mask_ids.append(id.replace('image', 'label'))
    masks = to_cropped_imgs(mask_ids, dir_mask, '.png', scale)
    masks_switched = map(hwc_to_chw, masks)

    return zip(imgs_normalized, masks_switched)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.png')
    mask = Image.open(dir_mask + id + '_mask.png')
    return np.array(im), np.array(mask)
