import torch
import sys
import gc
from torch.autograd import Variable


### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pyclipper
from shapely.geometry import Polygon
from charnet.modeling.postprocessing import load_char_dict
from pympler import muppy, summary

def generalized_dice_loss_w(y_true, y_pred): 
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    Ncl = y_pred.shape[-1]
    w = torch.zeros((Ncl,))
    for l in range(0,Ncl): w[l] = torch.sum( torch.asarray(y_true[:,:,:,:,l]==1,torch.int8) )
    w = 1/(w**2+0.00001)

    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*torch.sum(numerator,(0,1,2,3))
    numerator = torch.sum(numerator)
    
    denominator = y_true+y_pred
    denominator = w*torch.sum(denominator,(0,1,2,3))
    denominator = torch.sum(denominator)
    
    gen_dice_coef = numerator/denominator
    
    return 1-2*gen_dice_coef




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
        from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot1 = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot1[:, 0:1, :, :]
        true_1_hot_s = true_1_hot1[:, 1:2, :, :]
        true_1_hot2 = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)] #add class dimension by using eye
        true_1_hot2 = true_1_hot.permute(0, 3, 1, 2).float()  #move class to dimension 1
        probas = logits
        #probas = F.softmax(logits, dim=1) # suppress valuse below 1
        #torch.where(probas>0.2, probas, 0.0)
    true_1_hot3 = true_1_hot2.type(logits.type()) #Change type to floatTensor
    dims = (0,) + tuple(range(2, true.ndimension()+1)) # decide sum area
    intersection = torch.sum(probas * true_1_hot3, dims)
    cardinality = torch.sum(probas + true_1_hot3, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    
    true_1_hot=None
    true_1_hot2=None
    true_1_hot3=None
    true_1_hot_f=None
    true_1_hot_s=None
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


def keep_ce_loss(
            pred_char_fg, pred_char_cls,
            score_map_mask, score_map_char
    ):
        ce=nn.CrossEntropyLoss()
        nloss=nn.NLLLoss()
        char_boxes_pic=[]
        char_scores_pic=[]
        score_map_keep_pic=[]
        
        loss=torch.zeros(score_map_mask.shape[0])
        
        for pidx in range(score_map_mask.shape[0]):
            
            char_keep_rows, char_keep_cols = np.where(score_map_mask[pidx] > 0)

            
            char_scores = torch.zeros(char_keep_rows.shape[0], pred_char_cls.shape[1])
            score_map_char_keep = np.zeros(char_keep_rows.shape[0], dtype=np.int64)
        
            for idx in range(char_keep_rows.shape[0]):
                y, x = char_keep_rows[idx], char_keep_cols[idx]
                
                score = pred_char_fg[pidx][1][y, x]
                char_scores[idx, :] = pred_char_cls[pidx][:, y, x]
                score_map_char_keep [idx]=score_map_char[pidx][y,x]
            #keep, new_oriented_char_bboxes, new_char_scores = nms_with_char_cls_torch(
            #    oriented_char_bboxes, char_scores, self.char_nms_iou_thresh, num_neig=1
            #    )
            print("Char keep len = ", len(char_keep_rows))
        
      
            char_scores_pic.append(char_scores)
            score_map_keep_pic.append(score_map_char_keep)
            loss[pidx]=nloss(torch.log(char_scores), torch.from_numpy(score_map_char_keep.astype('int64')))
            char_scores=None
            score_map_char_keep=None
            

        return torch.mean(loss)


class char_matchingV2(nn.Module):
    def __init__(self, cfg):
        super(char_matching, self).__init__()
        self.char_dict_reverse = {}
        self.char_dict = load_char_dict(cfg.CHAR_DICT_FILE)
        self.num_class = len(self.char_dict)
        for k, v in self.char_dict.items():
            self.char_dict_reverse[v] = k 
        #self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.loss = nn.CrossEntropyLoss()
        self.debug = False
        
        
    def forward(self, char_bboxes, char_scores, polygon_chars, line_chars):
    
        
        match_all=[]
        loss=torch.tensor(0)
        rec_correct=0
        total_number=0
        
        if (self.debug):
            print("loss beginning")
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)
        
        for pic_idx in range(len(polygon_chars)):       #batch picture index
            maxchar_score = torch.argmax(char_scores[pic_idx], 1).cpu().numpy()  # get the highest score index
            if(len(char_scores[pic_idx])==0):print('empty char score');continue
            
            match_idx=np.empty(len(polygon_chars[pic_idx]))
            match_idx[:]=None
            for gidx in range(len(polygon_chars[pic_idx])):           #ground truth box
                inter_area=np.zeros(len(char_bboxes[pic_idx]))
                
                
                ppoly=pyclipper.scale_to_clipper(polygon_chars[pic_idx][gidx].reshape((4, 2)))
                if ((polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][1]).all() or
                    (polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][2]).all() or
                    (polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][1] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][2] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][1] == polygon_chars[pic_idx][gidx][2]).all()):
                    continue
            
                    
                if (self.debug==True): print('PPoly', polygon_chars[pic_idx][gidx])
                for pidx in range(len(char_bboxes[pic_idx])):           #predict box    
                    gpoly=pyclipper.scale_to_clipper(char_bboxes[pic_idx][pidx][:8].reshape((4, 2)))
                    gpoly1=char_bboxes[pic_idx][pidx][:8].reshape(4,2)
                    if (self.debug==True): print('GPoly', gpoly1)
                    if((gpoly1[0]==gpoly1[1]).all() or (gpoly1[0]==gpoly1[2]).all() or
                       (gpoly1[0]==gpoly1[3]).all() or (gpoly1[1]==gpoly1[3]).all() or
                       (gpoly1[2]==gpoly1[3]).all() or (gpoly1[1]==gpoly1[2]).all()):
                        solution=[]
                        inter=0
                        pc=None
                    
                    else:
                        pc = pyclipper.Pyclipper()
                        #print('PPoly', ppoly)
                        pc.AddPath(ppoly, pyclipper.PT_CLIP, True)
                        pc.AddPaths([gpoly], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                    
                    
                    if len(solution) == 0:
                        inter = 0
                        inter_area[pidx]=inter
                    else:
                        inter = pyclipper.scale_from_clipper(           #IOU calculation
                            pyclipper.scale_from_clipper(pyclipper.Area(solution[0])))
                        
                        inter_area[pidx]=inter
                     
                gmax_idx= np.argmax(inter_area) #Find the most match prdict box's index, inter_area len=predict box 

                if(inter_area[gmax_idx] != 0):
                    match_idx[gidx] = gmax_idx
            
                    
            if(self.debug):print (match_idx)
            pred_line=[]
            pred_number=[]
            for gidx in range(len(polygon_chars[pic_idx])):
                pidx_char = None
                if(not np.isnan(match_idx[gidx])):
                    if(self.debug):
                        print('gidx = ', gidx, "match_idx[gidx] = ", match_idx[gidx])
                        print("Maxcharscore ", maxchar_score[match_idx[gidx].astype('uint')] )
                    pred_line_chars= self.char_dict[maxchar_score[match_idx[gidx].astype('uint')].astype('uint')]
                    golden_char=line_chars[pic_idx][gidx]
                    pred_line.append(pred_line_chars)
                    pred_number.append(maxchar_score[match_idx[gidx].astype('int')].astype('int'))
                else:
                    pred_line.append(" ")
                    pred_number.append(-1)
            
            golden_class=[]
            for k in line_chars[pic_idx]:
                golden_class.append(self.char_dict_reverse[k.upper()])     
            golden_class_onehot=torch.eye(self.num_class)[golden_class].cuda()
            
            pred_class_score=char_scores[pic_idx][match_idx.astype('uint8').tolist()]
            
            indices=np.where (np.array(pred_number) == -1)[0]
            pred_class_score[indices] = 0
            loss=loss + self.loss(pred_class_score.cuda(), golden_class_onehot)
            for test, gold in zip(golden_class, pred_number):
                total_number = total_number +1
                if (test == gold):
                   rec_correct = rec_correct + 1
                   
                   
            #pred_class_onehot = torch.eye(self.num_class)[pred_number]
            #pred_class_onehot[indices] = 0
            #match_all.append(match_idx)
            match_idx=None
            pred_line=None 
            pred_number=None 
            inter_area=None
            gmax_idx=None
            pc=None
            golden_class=None
            dmaxchar_score=None

        
#        if 'golden_class_onehot' in locals():
#            del golden_class_onehot
#            del pred_class_score
#        gc.collect()
#        torch.cuda.empty_cache()
        
        loss=loss/len(polygon_chars)
        #loss=loss/self.num_class
        if(self.debug):
            print("polygon_chars ref count", sys.getrefcount(polygon_chars))
            print("loss ref count", sys.getrefcount(loss))
            print("loss end")
        
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)
        
#        if(self.debug):print(match_all)
#        del match_all
           
       
        return loss, total_number, rec_correct
#        return 0.1, 2, 1




class char_matching(nn.Module):
    def __init__(self, cfg):
        super(char_matching, self).__init__()
        self.char_dict_reverse = {}
        self.char_dict = load_char_dict(cfg.CHAR_DICT_FILE)
        self.num_class = len(self.char_dict)
        for k, v in self.char_dict.items():
            self.char_dict_reverse[v] = k 
        #self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.loss = nn.CrossEntropyLoss()
        self.debug = False
        
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
        
    def forward(self, char_bboxes, char_scores, polygon_chars, line_chars):
    
        
        match_all=[]
        loss=torch.tensor(0)
        rec_correct=0
        total_number=0
        
        if (self.debug):
            print("loss beginning")
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)
        
        for pic_idx in range(len(polygon_chars)):       #batch picture index
            maxchar_score = torch.argmax(char_scores[pic_idx], 1).cpu().numpy()  # get the highest score index
            if(len(char_scores[pic_idx])==0):print('empty char score');continue
            
            match_idx=np.empty(len(polygon_chars[pic_idx]))
            match_idx[:]=None
            for gidx in range(len(polygon_chars[pic_idx])):           #ground truth box
                inter_area=np.zeros(len(char_bboxes[pic_idx]))
                
                
                ppoly=pyclipper.scale_to_clipper(polygon_chars[pic_idx][gidx].reshape((4, 2)))
                if ((polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][1]).all() or
                    (polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][2]).all() or
                    (polygon_chars[pic_idx][gidx][0] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][1] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][2] == polygon_chars[pic_idx][gidx][3]).all() or
                    (polygon_chars[pic_idx][gidx][1] == polygon_chars[pic_idx][gidx][2]).all()):
                    continue
            
                    
                if (self.debug==True): print('PPoly', polygon_chars[pic_idx][gidx])
                for pidx in range(len(char_bboxes[pic_idx])):           #predict box    
                    gpoly=pyclipper.scale_to_clipper(char_bboxes[pic_idx][pidx][:8].reshape((4, 2)))
                    gpoly1=char_bboxes[pic_idx][pidx][:8].reshape(4,2)
                    if (self.debug==True): print('GPoly', gpoly1)
                    if((gpoly1[0]==gpoly1[1]).all() or (gpoly1[0]==gpoly1[2]).all() or
                       (gpoly1[0]==gpoly1[3]).all() or (gpoly1[1]==gpoly1[3]).all() or
                       (gpoly1[2]==gpoly1[3]).all() or (gpoly1[1]==gpoly1[2]).all()):
                        solution=[]
                        inter=0
                        pc=None
                    
                    else:
                        pc = pyclipper.Pyclipper()
                        #print('PPoly', ppoly)
                        pc.AddPath(ppoly, pyclipper.PT_CLIP, True)
                        pc.AddPaths([gpoly], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                    
                    
                    if len(solution) == 0:
                        inter = 0
                        inter_area[pidx]=inter
                    else:
                        inter = pyclipper.scale_from_clipper(           #IOU calculation
                            pyclipper.scale_from_clipper(pyclipper.Area(solution[0])))
                        
                        inter_area[pidx]=inter
                     
                gmax_idx= np.argmax(inter_area) #Find the most match prdict box's index, inter_area len=predict box 

                if(inter_area[gmax_idx] != 0):
                    match_idx[gidx] = gmax_idx
            
                    
            if(self.debug):print (match_idx)
            pred_line=[]
            pred_number=[]
            for gidx in range(len(polygon_chars[pic_idx])):
                pidx_char = None
                if(not np.isnan(match_idx[gidx])):
                    if(self.debug):
                        print('gidx = ', gidx, "match_idx[gidx] = ", match_idx[gidx])
                        print("Maxcharscore ", maxchar_score[match_idx[gidx].astype('uint')] )
                    pred_line_chars= self.char_dict[maxchar_score[match_idx[gidx].astype('uint')].astype('uint')]
                    golden_char=line_chars[pic_idx][gidx]
                    pred_line.append(pred_line_chars)
                    pred_number.append(maxchar_score[match_idx[gidx].astype('int')].astype('int'))
                else:
                    pred_line.append(" ")
                    pred_number.append(-1)
            
            golden_class=[]
            for k in line_chars[pic_idx]:
                golden_class.append(self.char_dict_reverse[k.upper()])     
            golden_class_onehot=torch.eye(self.num_class)[golden_class].cuda()
            
            pred_class_score=char_scores[pic_idx][match_idx.astype('uint8').tolist()]
            
            indices=np.where (np.array(pred_number) == -1)[0]
            pred_class_score[indices] = 0
            loss=loss + self.loss(pred_class_score.cuda(), golden_class_onehot)
            for test, gold in zip(golden_class, pred_number):
                total_number = total_number +1
                if (test == gold):
                   rec_correct = rec_correct + 1
                   
                   
            #pred_class_onehot = torch.eye(self.num_class)[pred_number]
            #pred_class_onehot[indices] = 0
            #match_all.append(match_idx)
            match_idx=None
            pred_line=None 
            pred_number=None 
            inter_area=None
            gmax_idx=None
            pc=None
            golden_class=None
            dmaxchar_score=None

        
#        if 'golden_class_onehot' in locals():
#            del golden_class_onehot
#            del pred_class_score
#        gc.collect()
#        torch.cuda.empty_cache()
        
        loss=loss/len(polygon_chars)
        #loss=loss/self.num_class
        if(self.debug):
            print("polygon_chars ref count", sys.getrefcount(polygon_chars))
            print("loss ref count", sys.getrefcount(loss))
            print("loss end")
        
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)
        
#        if(self.debug):print(match_all)
#        del match_all
           
       
        return loss, total_number, rec_correct
#        return 0.1, 2, 1
    
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
        #   d1 = Top, d2 = Right , d3 = Bottom, d4 = Left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(bbox_g, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(bbox_p, 1, 1)
    
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right

        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
        w_union = torch.min(d4_gt, d3_pred) + torch.min(d2_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d2_pred)
        I = w_union * h_union
        U = area_gt + area_pred - I
    
        w_enclose = torch.max(d4_gt, d3_pred) + torch.max(d2_gt, d4_pred)
        h_enclose = torch.max(d1_gt, d1_pred) + torch.max(d3_gt, d2_pred)
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
        #   d1 = Top, d2 = Right , d3 = Bottom , d4 = Left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(bbox_g, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(bbox_p, 1, 1)
    
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right

        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    
        w_union = torch.min(d4_gt, d3_pred) + torch.min(d2_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d2_pred)
        I = w_union * h_union
        U = area_gt + area_pred - I
    
        # calc area of Bc

        #U = area_pred + area_gt - I
        #iou = 1.0 * I / U

        return I, U, theta_pred, theta_gt    
    

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01
        #classification_loss *= 0.1

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        #   d1 = Top, d2 = Bottom, d3 = Left, d4 = Right
        
        #d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        #d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        #area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        #area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
        #w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        #h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        #area_intersect = w_union * h_union
        #area_union = area_gt + area_pred - area_intersect
        
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
        #return torch.mean(L_g.squeeze(1) * y_true_cls * training_mask) + classification_loss
        return torch.sum(L_g.squeeze(1) * y_true_cls * training_mask)/torch.sum(y_true_cls * training_mask) + classification_loss

