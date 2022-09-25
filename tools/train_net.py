# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from charnet.modeling.model import CharNet
import cv2, os
import numpy as np
from torch import nn
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt
from data.image_dataset import ImageDataset
from data.synth_dataset import SynthDataset
from data.data_loader import DataLoader, TrainSettings

from concern.config import Configurable, Config
from data.data_utils import generate_rbox
from data.data_loader import default_collate
from torch.optim import lr_scheduler
from tools.loss import *

def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    with open('{}/{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write('\n')


def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins

def drawPolys(image, poly, color):
    
    for i in range(len(poly)):
        pts=poly[i].astype('int')
        cv2.polylines(image, [pts], True, (0, 255,255))


def params_gen( net):
    
    params=[]
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{
                    'params': [value],
                    'lr': 0.001,
                    'weight_decay':0.0001}]
            else:
                params += [{
                    'params': [value],
                    'lr': 0.001,
                    'weight_decay': 0.0001
                }]
    return params
def train_model( args, cfg, img_loader ):
    

    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.train()
    charnet.cuda()
    params=params_gen(charnet)
    
    optimizer = torch.optim.SGD(params, momentum=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    criterion = LossFunc()
    
    #sequence=iter(img_loader)
    #images, polygons, polygon_chars, lines_texts, lines_chars, gts, ks, gt_chars, mask_chars, thresh_maps, thresh_masks, thresh_map_chars, thresh_mask_chars=next(sequence)
    #batch=next(sequence)
    #default_collate(batch)

    
#    for batch in img_loader:
#        for ind in range(len(batch['image'])):


#    for (images, polygons, polygon_chars, lines_texts, lines_chars, gts, ks, gt_chars, mask_chars) in img_loader:
    for (images, score_map, geo_map, training_mask, score_map_char, geo_map_char, training_mask_char, images_np) in img_loader: 
        char_bboxes, char_scores, word_instances, pred_word_fg, pred_word_tblr,\
            pred_word_orient, pred_char_fg, pred_char_tblr, pred_char_orient, pred_char_cls \
            = charnet(images.cuda(), 1, 1, images[0].size()[1], images[0].size()[2], images_np)
            
            
        #pred_word_tblr = torch.permute(pred_word_tblr, (0, 2, 3, 1))
        #pred_char_tblr = torch.permute(pred_char_tblr, (0, 2, 3, 1))
        char_size = pred_char_tblr.size()
        #pred_word_orient = torch.permute(pred_word_orient, (0, 2, 3, 1))
        pred_char_orient = torch.zeros([char_size[0], 1, char_size[2], char_size[3]]).cuda()
        
        pred_word_tblra = torch.cat((pred_word_tblr, pred_word_orient), 1)
        pred_char_tblra = torch.cat((pred_char_tblr, pred_char_orient), 1)
        
        pred_word_fg_sq = torch.squeeze(pred_word_fg[:,1::], 1)
        pred_char_fg_sq = torch.squeeze(pred_char_fg[:,1::], 1)
        
        score_map_char_mask=torch.where(score_map_char> 0, 1, 0)  
        score_map_char_adjust=torch.where(score_map_char> 0, -1, 0) 
        score_map_char = score_map_char + score_map_char_adjust
        
        score_map=score_map.cuda()
        geo_map=torch.permute(geo_map.cuda(), (0,3,1,2))
        training_mask=training_mask.cuda()
        score_map_char_mask=score_map_char_mask.cuda()
        score_map_mask=score_map_char.cuda()
        geo_map_char=torch.permute(geo_map_char[0].cuda(), (0,3,1,2))
        training_mask_char=training_mask_char.cuda()    
        
        
        loss1 = criterion(score_map, pred_word_fg_sq, geo_map, pred_word_tblra, training_mask)
        loss2 = criterion(score_map_char_mask, pred_char_fg_sq, geo_map_char, pred_char_tblra, training_mask_char)
        loss3 = dice_loss(score_map_char, pred_char_cls)
 
        loss_all = loss1 + loss2 + loss3
        print ("Loss all: ", loss_all, "loss1: ", loss1, "loss2: ", loss2, "loss3: ", loss3)
        #scheduler.step()
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        scheduler.step()

        #for ind in range(len(images)):
        #    char_bboxes, char_scores, word_instances = charnet(images[ind], 1, 1, images[ind].size()[0], images[ind].size()[1])
#            char_bboxes, char_scores, word_instances = charnet(batch['image'][ind].numpy().astype('uint8'), 1, 1, batch['image'][ind].size()[0], batch['image'][ind].size()[1])
        #    print(char_bboxes, word_instances)
        
        
        #Check  Image size. and transofrmation reuslt
#        print("Processing {}...".format(im_name))
#        im_file = os.path.join(args.image_dir, im_name)
#        im_original = cv2.imread(im_file)
#        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
#        with torch.no_grad():b
#            char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
#            save_word_recognition(
#                word_instances, os.path.splitext(im_name)[0],
#                args.results_dir, cfg.RESULTS_SEPARATOR
#            )

    
    
    
    
    
    return optimizer, params


def test_model(args, cfg):
    
    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.cuda()

    for im_name in sorted(os.listdir(args.image_dir)):
        print("Processing {}...".format(im_name))
        im_file = os.path.join(args.image_dir, im_name)
        im_original = cv2.imread(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
            save_word_recognition(
                word_instances, os.path.splitext(im_name)[0],
                args.results_dir, cfg.RESULTS_SEPARATOR
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("config_file", help="path to config file", type=str)
    parser.add_argument("image_dir", type=str)
    parser.add_argument("results_dir", type=str)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print(cfg)
    
    conf = Config()
    Trainsetting_conf=conf.compile(conf.load('./configs/seg_base.yaml'))
    Trainsetting_conf.update(cmd=args)
    
    cmd_in = vars(Trainsetting_conf.pop('cmd', dict()))
    cmd_in.update(is_train=True)
    train_cfg=Trainsetting_conf['Experiment']['train']
    train_cfg.update(cmd=cmd_in)
 
    train_synth_cfg=Trainsetting_conf['Experiment']['train_synth']
    train_synth_cfg.update(cmd=cmd_in)
    
    train_img_loader = Configurable.construct_class_from_config(train_cfg)
    train_synth_img_loader = Configurable.construct_class_from_config(train_synth_cfg)
    
    
    
    myprocess=train_synth_img_loader.data_loader.dataset.processes
    data = {}
    image_path=train_synth_img_loader.data_loader.dataset.image_paths[1]
    target = train_synth_img_loader.data_loader.dataset.targets[1]
    target_char = train_synth_img_loader.data_loader.dataset.targets_char[1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    data['filename']=image_path
    data['data_id']=image_path
    data['image']=img
    data['lines']=target
    data['chars']=target_char
    data1=myprocess[0](data)
    data2=myprocess[1](data1)
    data3=myprocess[2](data2)
    data4=myprocess[3](data3)
    data5=myprocess[4](data4)
    #data6=myprocess[5](data5)
    
    #dlen = len(train_synth_img_loader.data_loader)
    #print ('Len = ', dlen)
    
    #for i in range(dlen):
    #    temp=train_synth_img_loader.data_loader.dataset[i]
    #    print('image len =', len(temp['image']))
    
    
    
    
    
#    score_map, geo_map, training_mask, rects=generate_rbox((data6['image'].shape[1], data6['image'].shape[0]), data6['polygons'], data6['lines_text'])
    #data7 = myconv(data6['image'])
    #data7=myprocess[6](data6)
    
    #0 <data.processes.augment_data.AugmentDetectionData >, 
    #1 <data.processes.random_crop_data.RandomCropData >, 
    #2 <data.processes.make_icdar_data.MakeICDARData >, 
    #3 <data.processes.make_seg_detection_data.MakeSegDetectionData >, remove 
    #4->3 <data.processes.make_border_map.MakeBorderMap >, 
    #5->4 <data.processes.normalize_image.NormalizeImage >, 
    #6->5 <data.processes.filter_keys.FilterKeys >
    
    #org_image = data2['image'].copy()
    #print("Polys:", data2['polys'])
    #for i in range(len(data2['polys'])):item=data2['polys'][i]; pts=np.array(item['points']).astype('int');text=item['text'];cv2.polylines(org_image, [pts], True, (0, 255,255));cv2.putText(org_image, str(i) ,(pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    #for i in range(len(data2['polys_char'])):item=data2['polys_char'][i]; pts=np.array(item['points']).astype('int');text=item['text'];cv2.polylines(org_image, [pts], True, (0, 255,255));
    #cv2.imshow('TEST', org_image.astype('uint8'))
    #cv2.waitKey(0)
    
    #alpha = 0.5
    #beta = ( 1.0 - alpha );
    #dst=np.zeros(data5['image'].shape[:2], dtype=np.float32)
    #scorergb = np.dstack((data5['score_map']*255, data5['score_map']*255, data5['score_map']*255))
    #dst=cv2.addWeighted( data5['image'], alpha, scorergb.astype('float32'), beta, 0.0, dst);
    #cv2.imshow('TEST', dst.astype('uint8'))
    #cv2.waitKey(0)
    
    
    #Trainsetting(Trainsetting_conf['Experiment']['train']])

#Train code
    train_model(args, cfg, train_synth_img_loader.data_loader)

#TEST code
    test_model(args, cfg)
