import torch
from torch.autograd import Variable


### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn
import torch.nn.functional as F




def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
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
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)




def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection =torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss


def Giou_np(bbox_p, bbox_g):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(bbox_g, 1, 1)
    #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(bbox_p, 1, 1)
    
#    area_g = (d1_gt + d3_gt) * (d2_gt + d4_gt)
#    area_p = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    
#    x1p = torch.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
#    x2p = torch.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
#    y1p = torch.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
#    y2p = torch.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)

#    bbox_p = torch.cat((x1p, y1p, x2p, y2p), 1)
    # calc area of Bg
#    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    # calc area of Bp
#    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

    # cal intersection
    #d1I=torch.min(torch.cat((d1_gt, d1_pred), 1), 1).values.reshape(-1, 1)
    #d2I=torch.min(torch.cat((d2_gt, d2_pred), 1), 1).values.reshape(-1, 1)
    #d3I=torch.min(torch.cat((d3_gt, d3_pred), 1), 1).values.reshape(-1, 1)
    #d4I=torch.min(torch.cat((d4_gt, d4_pred), 1), 1).values.reshape(-1, 1)
    #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right

    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    I = w_union * h_union
    U = area_gt + area_pred - I
    
    w_enclose = torch.max(d3_gt, d3_pred) + torch.max(d4_gt, d4_pred)
    h_enclose = torch.max(d1_gt, d1_pred) + torch.max(d2_gt, d2_pred)
    area_c = w_enclose * h_enclose
    
    #x1I = torch.maximum(bbox_p[:, 0], bbox_g[:, 0])
    #y1I = torch.maximum(bbox_p[:, 1], bbox_g[:, 1])
    #x2I = torch.minimum(bbox_p[:, 2], bbox_g[:, 2])
    #y2I = torch.minimum(bbox_p[:, 3], bbox_g[:, 3])
    #I = torch.maximum((y2I - y1I), 0) * torch.maximum((x2I - x1I), 0)

    # find enclosing box
    #x1C = torch.minimum(bbox_p[:, 0], bbox_g[:, 0])
    #y1C = torch.minimum(bbox_p[:, 1], bbox_g[:, 1])
    #x2C = torch.maximum(bbox_p[:, 2], bbox_g[:, 2])
    #y2C = torch.maximum(bbox_p[:, 3], bbox_g[:, 3])

    # calc area of Bc
    #area_c = (x2C - x1C) * (y2C - y1C)
    U = area_pred + area_gt - I
    iou = 1.0 * I / U

    # Giou
    giou = iou - (area_c - U) / area_c

    # loss_iou = 1 - iou loss_giou = 1 - giou
    loss_iou = 1.0 - iou
    loss_giou = 1.0 - giou
    return giou, loss_iou, loss_giou



class LossFunc(nn.Module):
    def __init__(self, losstype='iou'):
        super(LossFunc, self).__init__()
        self.losstype = losstype
        return 
    
    
    def giou(self, bbox_p, bbox_g):
        """
        :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(bbox_g, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(bbox_p, 1, 1)
    
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        I = w_union * h_union
        U = area_gt + area_pred - I
    
        w_enclose = torch.max(d3_gt, d3_pred) + torch.max(d4_gt, d4_pred)
        h_enclose = torch.max(d1_gt, d1_pred) + torch.max(d2_gt, d2_pred)
        area_c = w_enclose * h_enclose

        # calc area of Bc
        #area_c = (x2C - x1C) * (y2C - y1C)
        U = area_pred + area_gt - I

        
        iou = 1.0 * I / U
        # Giou
        giou = iou - (area_c - U) / area_c

        # loss_iou = 1 - iou loss_giou = 1 - giou
        loss_iou = 1.0 - iou
        loss_giou = 1.0 - giou
        return I, U, area_c, theta_pred, theta_gt

    def iou(self, bbox_p, bbox_g):
        """
        :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(bbox_g, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(bbox_p, 1, 1)
    
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        I = w_union * h_union
        U = area_gt + area_pred - I
    
        # calc area of Bc

        U = area_pred + area_gt - I
        #iou = 1.0 * I / U

        return I, U, theta_pred, theta_gt    
    

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        
        if(self.losstype=='iou'):
            area_intersect, area_union, theta_pred, theta_gt = self.iou(y_pred_geo, y_true_geo)
            L_AABB = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
        else:    
            area_intersect, area_union, area_c, theta_pred, theta_gt = self.giou(y_pred_geo, y_true_geo)
            iou = 1.0 * area_intersect / area_union
            giou = iou - (area_c - area_union) / area_c
            L_AABB = 1.0 - giou
            
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return torch.mean(L_g * y_true_cls * training_mask) + classification_loss

