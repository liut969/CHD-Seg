import torch
from tqdm import tqdm
import numpy as np
from medpy.metric.binary import hd, dc
from utils import batch, plot_img_and_mask
import nibabel as nib
import re
import cv2
import os

# def normalize(image):
#     MIN_BOUND = np.min(image)
#     MAX_BOUND = np.max(image)
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image > 1] = 1.
#     image[image < 0] = 0.
#     return image

def eval_net(net, dataset, device, n_val):
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        img = b[0]
        true_mask = b[1][0]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask)

        img = img.to(device=device)
        true_mask = true_mask.to(device=device)

        mask_pred = net(img).squeeze(dim=0)

        mask_pred = mask_pred.cpu().detach().numpy()
        true_mask = true_mask.cpu().detach().numpy()
        mask_pred = np.argmax(mask_pred, 0)
        # print(np.unique(true_mask), np.unique(mask_pred))

        if (len(np.unique(true_mask)) == 1 and np.unique(true_mask) == [0]):
            n_val -= 1
        else:
            current_dice = 0
            classes = np.unique(true_mask)
            for c in [1, 2, 3, 4, 5, 6, 7]:
                gt_c_i = np.copy(true_mask)
                gt_c_i[gt_c_i != c] = 0
                pred_c_i = np.copy(mask_pred)
                pred_c_i[pred_c_i != c] = 0
                gt_c_i = np.clip(gt_c_i, 0, 1)
                pred_c_i = np.clip(pred_c_i, 0, 1)
                dice = dc(gt_c_i, pred_c_i)
                current_dice += dice
            # print(current_dice / (len(classes)-1))
            tot += (current_dice / (len(classes)-1))
            # plot_img_and_mask(true_mask, mask_pred)

    return tot / n_val

def get_3d_image_and_label(dir_path, image_name):

    map_need_crop_images = {"ct_1016_image.nii.gz": [50, 30, 480, 450],
                            "ct_1070_image.nii.gz": [40, 0, 480, 430], "ct_1072_image.nii.gz": [0, 0, 512, 460],
                            "ct_1074_image.nii.gz": [0, 0, 512, 460],
                            "ct_1106_image.nii.gz": [40, 40, 430, 430], "ct_1111_image.nii.gz": [20, 0, 480, 440],
                            "ct_1116_image.nii.gz": [40, 40, 450, 450], "ct_1124_image.nii.gz": [50, 0, 470, 370],
                            "ct_1126_image.nii.gz": [120, 130, 430, 410],
                            }

    map_need_fill_images = {"ct_1004_image.nii.gz": [210, 0, 210, 0], "ct_1021_image.nii.gz": [130, 0, 130, 0],
                            "ct_1022_image.nii.gz": [40, 0, 40, 0], "ct_1027_image.nii.gz": [150, 0, 150, 0],
                            "ct_1070_image.nii.gz": [100, 0, 100, 0],
                            }

    image_path = os.path.join(dir_path, image_name)
    nimg = nib.load(image_path)
    img = nimg.get_data()
    img = img.astype('float')
    img[img > 2000] = 2000

    label_name = image_name.replace('image', 'label')
    image_path = os.path.join(dir_path, label_name)
    nimg = nib.load(image_path)
    lab = nimg.get_data()
    lab[lab > 7] = 0        #### some labels have values greater than 7

    result_img = np.zeros([512, 512, img.shape[2]], dtype='float')
    result_lab = np.zeros([512, 512, lab.shape[2]])
    for i in range(img.shape[2]):
        norm = img[:, :, i]
        crop_label = lab[:, :, i]

        if image_name in map_need_crop_images.keys():
            location = map_need_crop_images[image_name]
            norm = norm[location[0]:location[2], location[1]:location[3]]
            crop_label = crop_label[location[0]:location[2], location[1]:location[3]]

        if image_name in map_need_fill_images.keys():
            location = map_need_fill_images[image_name]
            norm = cv2.copyMakeBorder(norm, location[0], location[1], location[2], location[3], cv2.BORDER_CONSTANT, 0)
            crop_label = cv2.copyMakeBorder(crop_label, location[0], location[1], location[2], location[3], cv2.BORDER_CONSTANT, 0)

        norm = np.uint8(cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX))
        norm = cv2.equalizeHist(norm)
        # norm = normalize(img[:, :, i])
        norm = cv2.resize(norm, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        result_img[:, :, i] = norm / 255.0
        result_lab[:, :, i] = cv2.resize(crop_label, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    return result_img, result_lab

#### reference from: https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py
def eval_net_3D(net, test_image_names, device, dir_path, save_dir):
    net.eval()
    tot = []
    for image_name in test_image_names:
        img, lab = get_3d_image_and_label(dir_path, image_name)
        for i in range(lab.shape[2]):
            lab[:, :, i] = cv2.resize(lab[:, :, i], dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            current_image = img[:, :, i]
            current_image = torch.from_numpy(current_image)
            current_image = current_image.to(device=device)
            current_image = current_image.unsqueeze(dim=0)
            current_image = current_image.unsqueeze(dim=0)
            current_image = current_image.float()

            mask_pred = net(current_image).squeeze(dim=0)
            mask_pred = mask_pred.cpu().detach().numpy()

            mask_pred = np.argmax(mask_pred, 0)
            img[:, :, i] = mask_pred
            if save_dir is not None:
                num = re.findall('\d+', image_name)
                save_name = os.path.join(save_dir, str(num[0]).zfill(4) + '.' + str(i+1).zfill(4) + '.png')
                cv2.imwrite(save_name, mask_pred * 30)

        res = []
        for c in [1, 2, 3, 4, 5, 6, 7]:
            gt_c_i = np.copy(lab)
            gt_c_i[gt_c_i != c] = 0
            pred_c_i = np.copy(img)
            pred_c_i[pred_c_i != c] = 0

            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            dice = dc(gt_c_i, pred_c_i)

            res += [dice]
        print(image_name, res, sum(res)/len(res))
        tot += [res]
    tot = np.sum(tot, axis=0)
    tot = np.divide(tot, len(test_image_names))
    return tot
