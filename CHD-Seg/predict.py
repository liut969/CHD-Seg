import torch
from tqdm import tqdm
import numpy as np
from medpy.metric.binary import hd, dc
from net import UNet_GloRe
from eval import eval_net, eval_net_3D
from utils import get_imgs_and_masks, plot_img_and_mask, get_ids
import os
from sklearn.model_selection import KFold

def get_test_Dice():
    dir_path = '../dataset/CHD_segmentation_dataset'
    save_dir = '../dataset/CHD_predict'
    image_names = [f for f in os.listdir(dir_path) if f.endswith('image.nii.gz')]
    image_names = sorted(image_names)
    cross_validation = 4
    current_cross_validation = 1        ####0, 1, 2, 3
    save_dir = save_dir + '.' + str(current_cross_validation)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    kf = KFold(n_splits=cross_validation)
    for i, (train_idx, test_idx) in enumerate(kf.split(image_names)):
        if i == current_cross_validation:
            train_names = np.array(image_names)[train_idx]
            test_names = np.array(image_names)[test_idx]
            break
    print('test image names:', test_names)
    if current_cross_validation == 2:
        test_names = np.delete(test_names, [7, 8, 9, 12], axis=0)   #### dirty data
        print('test image names:', test_names)

    net = UNet_GloRe(n_channels=1, n_classes=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load('./checkpoints/CP_epoch20.pth', map_location=device))
    test_score = eval_net_3D(net, test_names, device, dir_path, save_dir)

    print('test_score:', test_score)
    print('mean:', sum(test_score)/len(test_score))


if __name__ == '__main__':
    get_test_Dice()



