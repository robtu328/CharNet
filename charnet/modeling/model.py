# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from charnet.modeling.backbone.resnet import resnet50
#from charnet.modeling.backbone.hourglassGCN import hourglass88
from charnet.modeling.backbone.hourglass import hourglass88, hourglass88GCN, Residual
#from charnet.modeling.backbone.hourglass import hourglass88
from charnet.modeling.backbone.decoder import Decoder
from collections import OrderedDict
from torch.functional import F
from charnet.modeling.layers import Scale
import torchvision.transforms as T
from .postprocessing import OrientedTextPostProcessing
from charnet.config import cfg
import numpy as np
from interimage.ops_dcnv3 import modules as opsm
#from interimage.intern_image import InternImage


def _conv3x3_bn_relu(in_channels, out_channels, dilation=1):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU())
    ]))


def to_numpy_or_none(*tensors):
    results = []
    for t in tensors:
        if t is None:
            results.append(None)
        else:
            results.append(t.cpu().numpy())
    return results


class WordDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, dilation=1):
        super(WordDetector, self).__init__()
        self.word_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels, dilation
        )
        self.word_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.word_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.word_det_conv_final(x)

        pred_word_fg = self.word_fg_pred(self.word_fg_feat(feat))

        word_regression_feat = self.word_regression_feat(feat)
        pred_word_tblr = F.relu(self.word_tblr_pred(word_regression_feat)) * 10.
        pred_word_orient = self.orient_pred(word_regression_feat)

        return pred_word_fg, pred_word_tblr, pred_word_orient


class CharDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, curved_text_on=False):
        super(CharDetector, self).__init__()
        self.character_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels
        )
        self.char_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.char_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)

    def forward(self, x):
        feat = self.character_det_conv_final(x)

        pred_char_fg = self.char_fg_pred(self.char_fg_feat(feat))
        char_regression_feat = self.char_regression_feat(feat)
        pred_char_tblr = F.relu(self.char_tblr_pred(char_regression_feat)) * 10.
        pred_char_orient = None

        return pred_char_fg, pred_char_tblr, pred_char_orient


class CharRecognizer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, num_classes):
        super(CharRecognizer, self).__init__()

        self.body = nn.Sequential(
            _conv3x3_bn_relu(in_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
        )
        self.classifier = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1)

        core_op=getattr(opsm, 'DCNv3_pytorch')

        self.dcn = nn.Sequential(
            core_op( channels=bottleneck_channels, kernel_size=3 , stride=1, pad=1,dilation=1,
                        group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN',
                        dw_kernel_size=None, center_feature_scale=False, imgFmt='CHW'),
            Residual(bottleneck_channels, num_classes, stride=1)    #channels in= 128, channels out=256
        ) 

    def forward(self, feat):
        feat = self.body(feat)
        return self.classifier(feat)
        #return self.dcn(feat)

class CharNet(nn.Module):
    def __init__(self, backbone=hourglass88()):
    #def __init__(self, backbone=hourglass88GCN()):

        super(CharNet, self).__init__()
        self.backbone = backbone
        decoder_channels = 256
        bottleneck_channels = 128
        self.word_detector = WordDetector(
            decoder_channels, bottleneck_channels,
            dilation=cfg.WORD_DETECTOR_DILATION
        )
        self.char_detector = CharDetector(
            decoder_channels,
            bottleneck_channels
        )
        self.char_recognizer = CharRecognizer(
            decoder_channels, bottleneck_channels,
            num_classes=cfg.NUM_CHAR_CLASSES
        )

        args = {
            "word_min_score": cfg.WORD_MIN_SCORE,
            "word_stride": cfg.WORD_STRIDE,
            "word_nms_iou_thresh": cfg.WORD_NMS_IOU_THRESH,
            "char_stride": cfg.CHAR_STRIDE,
            "char_min_score": cfg.CHAR_MIN_SCORE,
            "num_char_class": cfg.NUM_CHAR_CLASSES,
            "char_nms_iou_thresh": cfg.CHAR_NMS_IOU_THRESH,
            "char_dict_file": cfg.CHAR_DICT_FILE,
            "word_lexicon_path": cfg.WORD_LEXICON_PATH
        }

        self.post_processing = OrientedTextPostProcessing(**args)

        self.transform = self.build_transform()
        #self.interImage = InternImage(channels=256, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32],layer_scale=1.0)
        #print(self.interImage)

    def forward(self, im, im_scale_w, im_scale_h, original_im_w, original_im_h, images_np):
        #im = self.transform(im).cuda()
        #im=im.cuda()
        
        #im = im.cuda()
        #im = im.unsqueeze(0)
        
        #myconv = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        #myconv.cuda()
        #im1=im.clone()
        features = self.backbone(im)
        #features1 = self.interImage(im)

        #Internet
        #pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features[0])
        #pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features[0])
        #recognition_results = self.char_recognizer(features[0])
        
        #hourglass
        pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features)
        pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features)
        recognition_results = self.char_recognizer(features)

        pred_word_fg = F.softmax(pred_word_fg, dim=1)
        pred_char_fg = F.softmax(pred_char_fg, dim=1)
        pred_char_cls = F.softmax(recognition_results, dim=1)

        if pred_word_fg is None:
            pred_word_fg_np = pred_word_fg
        else:
            pred_word_fg_np = pred_word_fg.clone().detach()
        
        if pred_word_tblr is None:  
            pred_word_tblr_np = pred_word_tblr
        else:
            pred_word_tblr_np = pred_word_tblr.clone().detach()
            
        if pred_word_orient is None:  
            pred_word_orient_np = pred_word_orient
        else:    
            pred_word_orient_np = pred_word_orient.clone().detach()
            
        if pred_char_fg is None:  
            pred_char_fg_np = pred_char_fg
        else:  
            pred_char_fg_np = pred_char_fg.clone().detach()
            
        if pred_char_tblr is None:  
            pred_char_tblr_np = pred_char_tblr
        else:  
            pred_char_tblr_np = pred_char_tblr.clone().detach()
            
        if pred_char_cls is None:  
            pred_char_cls_np = pred_char_cls
        else:  
            pred_char_cls_np = pred_char_cls.clone().detach()
            
        if pred_char_orient is None:  
            pred_char_orient_np = pred_char_orient
        else:  
            pred_char_orient_np = pred_char_orient.clone().detach()

        pred_word_fg_np, pred_word_tblr_np, \
        pred_word_orient_np, pred_char_fg_np, \
        pred_char_tblr_np, pred_char_cls_np, \
        pred_char_orient_np = to_numpy_or_none(
            pred_word_fg_np, pred_word_tblr_np,
            pred_word_orient_np, pred_char_fg_np,
            pred_char_tblr_np, pred_char_cls_np,
            pred_char_orient_np
        )

        char_bboxes=[]
        char_scores=[]
        word_instances=[]
        valid_boxes=[]
        ss_word_bboxes=[]
        
        for idx in range(im.size()[0]):
            char_bboxe, char_score, word_instance, valid_boxe, ss_word_bboxe = self.post_processing(
                pred_word_fg_np[idx, 1], pred_word_tblr_np[idx],
                pred_word_orient_np[idx, 0], pred_char_fg_np[idx, 1],
                pred_char_tblr_np[idx], pred_char_cls[idx],
                im_scale_w, im_scale_h,
                original_im_w, original_im_h
            )
            
            #char_score_torch = self.post_processing.parse_char_torch(
            #    pred_word_fg_np[idx, 1], pred_char_fg_np[idx, 1], pred_char_tblr_np[idx], pred_char_cls[idx],
            #    im_scale_w, im_scale_h, original_im_w, original_im_h
            #)
            
            char_bboxes.append(char_bboxe)
            char_scores.append(char_score)
            word_instances.append(word_instance)
            valid_boxes.append(valid_boxe)
            ss_word_bboxes.append(ss_word_bboxe)
        #char_bboxe=None
        #char_score=None
        #word_instance=None
        #valid_boxe=None
        #ss_word_bboxe=None        
            
        return char_bboxes, char_scores, word_instances, pred_word_fg, pred_word_tblr, pred_word_orient, pred_char_fg, pred_char_tblr, pred_char_orient, pred_char_cls, ss_word_bboxes 

    def loss_cal():
        loss=1
        return loss
    def build_transform(self):
        to_rgb_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_rgb_transform,
                normalize_transform,
            ]
        )
        return transform


    def build_invtransform(self):
        to_rgb_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_rgb_transform,
                normalize_transform,
            ]
        )
        return transform
