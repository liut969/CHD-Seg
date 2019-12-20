import cv2
import random
import numpy as np

def random_flip(image, mask, random_rate=0.5):
    if random.random() < random_rate:
        flip_code = random.randint(-1, 1)
        image = cv2.flip(image, flip_code)
        mask = cv2.flip(mask, flip_code)
    return image, mask

def random_rotate(image, mask, random_rate=0.5):
    if random.random() < random_rate:
        rows, cols = image.shape[:2]
        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
        image = cv2.warpAffine(image, M, (rows, cols))
        mask = cv2.warpAffine(mask, M, (rows, cols))
    return image, mask


def random_scale_and_crop(image, mask, random_rate=0.5):
    #### It is assumed here that the length and width of the image are equal. E.g. [512, 512]
    if random.random() < random_rate:
        base_size = image.shape[0]
        scale_size = random.randint(int(base_size * 0.9), int(base_size * 1.1))
        image = cv2.resize(image, (scale_size, scale_size), cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (scale_size, scale_size), cv2.INTER_NEAREST)
        if scale_size < base_size:
            pad = base_size - scale_size
            image = cv2.copyMakeBorder(image, 0, pad, 0, pad, cv2.BORDER_CONSTANT, 0)
            mask = cv2.copyMakeBorder(mask, 0, pad, 0, pad, cv2.BORDER_CONSTANT, 0)
        else:
            crop_location = random.randint(0, scale_size - base_size)
            image = image[crop_location: crop_location+base_size, crop_location: crop_location+base_size]
            mask = mask[crop_location: crop_location+base_size, crop_location: crop_location+base_size]
    return image, mask



def random_data_augmentation(image_and_mask, random_rate=0.5):
    image_and_mask = np.squeeze(image_and_mask, axis=2)
    for i, current_image_and_mask in enumerate(image_and_mask):
        current_image = current_image_and_mask[0]
        current_mask = current_image_and_mask[1]
        # current_image, current_mask = random_scale_and_crop(current_image, current_mask, random_rate)
        # current_image, current_mask = random_flip(current_image, current_mask, random_rate)
        current_image, current_mask = random_rotate(current_image, current_mask, random_rate)

        image_and_mask[i][0] = current_image
        image_and_mask[i][1] = current_mask
    image_and_mask = np.expand_dims(image_and_mask, axis=2)
    return image_and_mask


