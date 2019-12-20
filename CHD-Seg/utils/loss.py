from torch.nn import functional as F
import torch
import numpy as np
import cv2


def soft_dice_loss(predict, soft_y, num_class, softmax = True):
    #### reference from: https://github.com/ihil/PyMIC/blob/eaeb496db257b6a58923ecdde25b25be00dd3fe4/pymic/test/test_tensorboard.py#L37
    #### https://stackoverflow.com/questions/44461772/creating-one-hot-vector-from-indices-given-as-a-tensor
    soft_y = torch.nn.functional.one_hot(soft_y, num_class)
    soft_y  = torch.reshape(soft_y, (-1, num_class))
    predict = predict.permute(0, 2, 3, 1)
    predict = torch.reshape(predict, (-1, num_class))
    if(softmax):
        predict = nn.Softmax(dim=-1)(predict)
    y_vol = torch.sum(soft_y, dim=0)
    p_vol = torch.sum(predict, dim=0)
    intersect = torch.sum(soft_y * predict, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    dice_score = torch.mean(dice_score)
    return 1.0 - dice_score


def dice_loss(true, logits, eps=1e-7):
    """
    reference from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    true = true.unsqueeze(1)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    # dice_loss = (2. * intersection / (cardinality + eps)).mean()
    dice_loss = (2. * intersection / (cardinality + eps))
    dice_loss = torch.mean(dice_loss)
    return (1 - dice_loss)

def shape_match_loss(true, logits, true_masks_np, eps=1e-7):
    bs = logits.shape[0]
    num_classes = logits.shape[1]
    true = true.unsqueeze(1)
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)                                    #####shape[bs, num_class, w, h]
    true_1_hot = true_1_hot.type(logits.type())

    pred_masks_np = probas.clone().data.cpu().numpy()
    pred_masks_np = np.argmax(pred_masks_np, axis=1)

    shape_loss = torch.autograd.Variable(torch.zeros(bs))
    for i in range(bs):
        current_intersection = torch.sum(probas[i] * true_1_hot[i], (1, 2))
        current_cardinality = torch.sum(probas[i] + true_1_hot[i], (1, 2))
        current_dice_loss = 1.0 - ((2.0 * current_intersection + eps) / (current_cardinality + eps))
        for j in range(num_classes):
            current_true_label_part = np.zeros(true.shape[-2:])
            current_true_label_part[true_masks_np[i] == j] = 1
            current_pred_label_part = np.zeros(true.shape[-2:])
            current_pred_label_part[pred_masks_np[i] == j] = 1
            similarity_label = cv2.matchShapes(current_true_label_part, current_pred_label_part, 1, 0)
            similarity_label = 1.0 / (1.0 + np.exp(-similarity_label))
            current_dice_loss[j] *= similarity_label
        current_dice_loss = torch.mean(current_dice_loss)
        shape_loss[i] = current_dice_loss
    result_loss = torch.mean(shape_loss)
    return result_loss

