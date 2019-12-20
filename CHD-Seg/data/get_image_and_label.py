import numpy as np
import os
import nibabel as nib
import cv2
import re
from tqdm import tqdm
from sklearn.model_selection import KFold

# def normalize(image):
#     MIN_BOUND = np.min(image)
#     MAX_BOUND = np.max(image)
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image > 1] = 1.
#     image[image < 0] = 0.
#     return image

def load_one_image(dir_path, image_name, label_name, save_image_path, save_label_path):

    image_path = os.path.join(dir_path, image_name)
    nimg = nib.load(image_path)
    img = nimg.get_data()   #### [w, h, d]
    img = img.astype('float')
    img[img > 2000] = 2000

    image_path = os.path.join(dir_path, label_name)
    nimg = nib.load(image_path)
    lab = nimg.get_data()
    lab[lab > 7] = 0        #### some labels have values greater than 7

    for i in range(img.shape[2]):
        num = re.findall('\d+', image_name)
        save_name = os.path.join(save_image_path, str(num[0]).zfill(4) + '.' + str(i+1).zfill(4) + '.png')
        norm = img[:, :, i]
        crop_label = lab[:, :, i]
        norm = np.uint8(cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX))
        norm = cv2.equalizeHist(norm)
        # norm = normalize(img[:, :, i])
        norm = cv2.resize(norm, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_name, norm)

        num = re.findall('\d+', label_name)
        save_name = os.path.join(save_label_path, str(num[0]).zfill(4) + '.' + str(i+1).zfill(4) + '.png')
        cv2.imwrite(save_name, cv2.resize(crop_label, dsize=(512, 512), interpolation=cv2.INTER_NEAREST))


def load_all_image(dir_path):
    image_names = [f for f in os.listdir(dir_path) if f.endswith('image.nii.gz')]
    image_names = sorted(image_names)
    cross_validation = 4
    current_cross_validation = 1        ####0, 1, 2, 3
    kf = KFold(n_splits=cross_validation)
    for i, (train_idx, test_idx) in enumerate(kf.split(image_names)):
        if i == current_cross_validation:
            train_names = np.array(image_names)[train_idx]
            test_names = np.array(image_names)[test_idx]
            break
    print('cross_validation:', cross_validation)
    print('current_cross_validation:', current_cross_validation)
    print('train_names:', train_names)
    print('test_names:', test_names)

    save_image_dir_name = 'CHD.image.' + str(current_cross_validation)
    if not os.path.isdir(os.path.join(os.path.dirname(dir_path), save_image_dir_name)):
        os.mkdir(os.path.join(os.path.dirname(dir_path), save_image_dir_name))
    save_image_path = os.path.join(os.path.dirname(dir_path), save_image_dir_name)

    save_label_dir_name = 'CHD.label.' + str(current_cross_validation)
    if not os.path.isdir(os.path.join(os.path.dirname(dir_path), save_label_dir_name)):
        os.mkdir(os.path.join(os.path.dirname(dir_path), save_label_dir_name))
    save_label_path = os.path.join(os.path.dirname(dir_path), save_label_dir_name)

    for f_name in tqdm(train_names):
        image_name = f_name
        label_name = f_name.replace('image', 'label')
        load_one_image(dir_path, image_name, label_name, save_image_path, save_label_path)

    save_image_dir_name = 'CHD.image.test.' + str(current_cross_validation)
    if not os.path.isdir(os.path.join(os.path.dirname(dir_path), save_image_dir_name)):
        os.mkdir(os.path.join(os.path.dirname(dir_path), save_image_dir_name))
    save_image_path = os.path.join(os.path.dirname(dir_path), save_image_dir_name)

    save_label_dir_name = 'CHD.label.test.' + str(current_cross_validation)
    if not os.path.isdir(os.path.join(os.path.dirname(dir_path), save_label_dir_name)):
        os.mkdir(os.path.join(os.path.dirname(dir_path), save_label_dir_name))
    save_label_path = os.path.join(os.path.dirname(dir_path), save_label_dir_name)

    for f_name in tqdm(test_names):
        image_name = f_name
        label_name = f_name.replace('image', 'label')
        load_one_image(dir_path, image_name, label_name, save_image_path, save_label_path)


def _main():
    print(os.getcwd())
    dir_path = '../../dataset/CHD_segmentation_dataset'
    load_all_image(dir_path)


if __name__ == '__main__':
    _main()
