# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import cv2, os, sys
sys.path.append(os.getcwd())

from charnet.modeling.model import CharNet
import numpy as np
from torch import nn
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt
from data.image_dataset import ImageDataset
from data.synth_dataset import SynthDataset
from data.data_loader import DataLoader, TrainSettings
#from interimage.intern_image import InternImage
from charnet.modeling.backbone.hourglass import hourglass88, hourglass88GCN

from concern.config import Configurable, Config
from data.data_utils import generate_rbox, blending_two_imgs, bonding_box_plane, createBlankPicture, draw_polys
from data.data_loader import default_collate
from torch.optim import lr_scheduler
from tools.loss import *
from GPUtil import showUtilization as gpu_usage
import psutil
from pympler import muppy, summary


#Profile
import gc
import cProfile
import pstats
from pstats import SortKey
import tracemalloc

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
            (word_bbox[0].astype('int32'), word_bbox[1].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return img_word_ins

def drawBoxes(img, boxes, color):
    img_word_ins = img.copy()
    for i in range(len(boxes)):
        pts=boxes[i][:8].reshape((-1, 2)).astype('int')
        cv2.polylines(img_word_ins, [pts], True, (0, 255,255))
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
                    #'lr': 0.001,
                    'lr': 0.005,
                    'weight_decay':0.0001
                }]
            else:
                params += [{
                    'params': [value],
                    #'lr': 0.001,
                    'lr': 0.005,
                    'weight_decay': 0.0001
                }]
    return params


def validate_model( charnet, args, cfg, img_loader, valid_cfg, iou_setting, debug=False):
       
    #debug=True
    
    charnet.eval()
    charnet.cuda()
    params=params_gen(charnet)
    #img_loader.dataset.mode = 'valid' 
    #img_loader.dataset.mode = 'train' 


    
    invTrans  = img_loader.dataset.processes[4]

    criterion_w = LossFunc() 
    criterion_c = LossFunc() 
    cmatch= char_matching(cfg)
    #cregloss=char_reg_loss(cfg)
    cregloss=char_reg_lossV2(cfg)
    char_dict = load_char_dict(cfg.CHAR_DICT_FILE)


    back_batch_time = 8
    batch_times = 0
    loss_all = 0
    
    time=0
    if debug:
        tracemalloc.start()
        snapshotO = tracemalloc.take_snapshot()
    
    for eidx in range (1):
        loss1_total = 0
        loss2_total = 0
        loss3_total = 0
        loss4_total = 0
        iter_cnt = 0
        total_number=1
        correct_number=0
        cnt_dict_showup = np.zeros(len(char_dict))
        cnt_dict_correct = np.zeros(len(char_dict))
        cnt_dict_gt = np.zeros(len(char_dict))
        
        debug = False
        for (images, score_map, geo_map, training_mask, score_map_char, geo_map_char, training_mask_char, images_np, polygon_chars, line_chars, indexes) in img_loader: 
            #image format : CHW
            char_bboxes, char_scores, word_instances, pred_word_fg, pred_word_tblr,\
                pred_word_orient, pred_char_fg, pred_char_tblr, pred_char_orient, pred_char_cls, ss_word_bboxes \
                    = charnet(images.cuda(), 1, 1, images[0].size()[2], images[0].size()[1], images_np)
                 
            
            
            char_size = pred_char_tblr.size()
            pred_char_orient = torch.zeros([char_size[0], 1, char_size[2], char_size[3]]).cuda()
        
            pred_word_tblra = torch.cat((pred_word_tblr, pred_word_orient), 1)
            pred_char_tblra = torch.cat((pred_char_tblr, pred_char_orient), 1)
            
            pred_word_fg_sq = torch.squeeze(pred_word_fg[:,1::], 1)
            pred_char_fg_sq = torch.squeeze(pred_char_fg[:,1::], 1)
           

            score_map_char_mask=torch.where(score_map_char> 0.0, torch.tensor(1.0, dtype=torch.float64), 0.0)  
            score_map_char_adjust=torch.where(score_map_char> 0.0, torch.tensor(-1.0, dtype=torch.float64), 0.0) 
            score_map_char = score_map_char + score_map_char_adjust
                        
            pred_word_fg_clip = torch.where(pred_word_fg_sq > 0.9, 1.0, 0.0)#.cpu().numpy()
            pred_char_fg_clip = torch.where(pred_char_fg_sq > 0.25, 1.0, 0.0)#.cpu().numpy()

            #debug_class  ground truth word class and char class 
            if valid_cfg['debug_class']:
                print("Image Path", img_loader.dataset.image_paths[indexes[0]])
                img=invTrans.lib_inv_trans(images[0])
                blend_img=blending_two_imgs(img, score_map[0].cpu().numpy().astype('uint8'))
                #blend_char_img=blending_two_imgs(img, score_map_char[0].cpu().numpy().astype('uint8'))
                #score_map_char_mask=torch.where(score_map_char> 0, 1, 0)
                blend_char_img=blending_two_imgs(img, score_map_char_mask[0].cpu().numpy().astype('uint8'))
                cv2.destroyAllWindows()
                cv2.imshow("score_map", blend_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                cv2.imshow("score_map_char", blend_char_img)
                cv2.waitKey()
                
            #debug_class  predict word class and char class 
            if valid_cfg['debug_class']:  # Predict word/char classes showing 
                img=invTrans.lib_inv_trans(images[0])
                blend_img=blending_two_imgs(img, pred_word_fg_clip[0].cpu().numpy().astype('uint8'), 0.5, 0.5)
                #cv2.destroyAllWindows()
                #cv2.imshow("pred_word_fg", blend_img)
                #cv2.waitKey()
                
                #blend_char_img=blending_two_imgs(img, pred_char_fg_clip[0].cpu().numpy().astype('uint8'))
                blend_char_img=blending_two_imgs(blend_img, pred_char_fg_clip[0].cpu().numpy().astype('uint8'), 0.5, 0.5)
                cv2.destroyAllWindows()
                cv2.imshow("pred_wordchar_fg", blend_char_img)
                cv2.waitKey()

            
            #img=invTrans.lib_inv_trans(images[0])
            #for i in range(len(valid_boxe)):b=valid_boxe[i][1][0:8].astype('int32'); pts=[b];cv2.polylines(img, pts, True, (0, 0,255));

            
            score_map=score_map.cuda()
            geo_map=torch.permute(geo_map.cuda(), (0,3,1,2))
            training_mask=training_mask.cuda()
            score_map_char_mask=score_map_char_mask.cuda()
            score_map_mask=score_map_char.cuda()
            score_map_char_mask_np=score_map_char_mask.cpu().numpy()
            
            geo_map_char=torch.permute(geo_map_char[0].cuda(), (0,3,1,2))
            training_mask_char=training_mask_char.cuda()    
                    
            
            

            
            # ground truth and Predict word/char boxes showing        
            if valid_cfg['debug_box']:
                img=invTrans.lib_inv_trans(images[0])
                ic, ih, iw= images[0].shape
                bbox_gt=bonding_box_plane(geo_map, 'ground')
                score_map_unsqueeze=torch.stack((score_map, score_map, score_map, score_map, score_map), axis=1)
                word_fg_unsqueeze=torch.stack((pred_word_fg_clip, pred_word_fg_clip, pred_word_fg_clip, pred_word_fg_clip, pred_word_fg_clip), axis=1)
                #bbox_pd=bonding_box_plane(pred_word_tblra*score_map_unsqueeze)
                bbox_pd=bonding_box_plane(pred_word_tblra*word_fg_unsqueeze)
                bbox_gt1 = cv2.resize(bbox_gt, (iw, ih), interpolation=cv2.INTER_AREA).astype('uint8')
                bbox_pd1 = cv2.resize(bbox_pd, (iw, ih), interpolation=cv2.INTER_AREA).astype('uint8')
                
                blend_pic= cv2.addWeighted( img, 0.5, bbox_gt1, 0.5, 0.0)
                cv2.destroyAllWindows()
                cv2.imshow('Word geo_map bbox_gt1', blend_pic)
                cv2.waitKey()
            
                blend_pic= cv2.addWeighted( img, 0.5, bbox_pd1, 0.5, 0.0)
                cv2.destroyAllWindows()
                cv2.imshow('Word pred_rsult bbox_pd1', blend_pic)
                cv2.waitKey()
            
                bbox_gt_char=bonding_box_plane(geo_map_char, 'ground')
                
                score_map_char_unsqueeze=torch.stack((score_map_char_mask, score_map_char_mask, score_map_char_mask, score_map_char_mask), axis=1)
                char_fg_unsqueeze=torch.stack((pred_char_fg_clip, pred_char_fg_clip, pred_char_fg_clip, pred_char_fg_clip), axis=1)
                #bbox_pd_char=bonding_box_plane(pred_char_tblr*score_map_char_unsqueeze)
                bbox_pd_char=bonding_box_plane(pred_char_tblr*char_fg_unsqueeze)
                bbox_gt_char1 = cv2.resize(bbox_gt_char, (iw, ih), interpolation=cv2.INTER_AREA).astype('uint8')
                bbox_pd_char1 = cv2.resize(bbox_pd_char, (iw, ih), interpolation=cv2.INTER_AREA).astype('uint8')
                blend_pic= cv2.addWeighted( img, 0.5, bbox_gt_char1, 0.5, 0.0)
                
                cv2.destroyAllWindows()
                cv2.imshow('Char geo_map bbox_gt_char1', blend_pic)
                cv2.waitKey()
            
                blend_pic= cv2.addWeighted( img, 0.5, bbox_pd_char1, 0.5, 0.0)
                cv2.destroyAllWindows()
                cv2.imshow('Char pred_rsult bbox_pd_char1', blend_pic)
                cv2.waitKey()



            loss1 = criterion_w(score_map, pred_word_fg_sq, geo_map, pred_word_tblra, training_mask)
            loss2 = criterion_c(score_map_char_mask, pred_char_fg_sq, geo_map_char, pred_char_tblra, training_mask_char)
            #loss3 = dice_loss(score_map_char, pred_char_cls*score_map_char_mask.unsqueeze(1))
            #loss3 = dice_loss(score_map_char, pred_char_cls*score_map_char_mask.unsqueeze(1))
            
            loss5 = keep_ce_loss(pred_char_fg, pred_char_cls, score_map_char_mask_np, score_map_char)

            #debug_out Final Predict word/char boxes showing
            if valid_cfg['debug_out']:
                #boxes_list = [data[1].astype('uint32') for data in ss_word_bboxes[0]]
                color = 0
                img=invTrans.lib_inv_trans(images[0])
                img_box = drawBoxes(img, ss_word_bboxes[0], color)
                cv2.destroyAllWindows()
                cv2.imshow('ss_word_bboxes', img_box)
                cv2.waitKey()
                
                img_box = drawBoxes(img, char_bboxes[0], color)
                cv2.destroyAllWindows()
                cv2.imshow('char_bboxes', img_box)
                cv2.waitKey()
                
                img_box=vis(img, word_instances[0])
                cv2.destroyAllWindows()
                cv2.imshow('word_instances', img_box)
                cv2.waitKey()
            
            if debug == True:
                print ("Memory check start")
                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)
            
            #loss4, number, correct = cmatch(char_bboxes, char_scores, polygon_chars, line_chars)
            number1, correct1, cnt_dict_show1, cnt_dict_correct1, cnt_dict_gt1 =  cregloss(word_instances, polygon_chars, line_chars, min_iou=iou_setting)
            loss3=0
            loss4=0 
            #number=1
            #correct=0
            
            
            if debug == True:
                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)
                print ("Memory check end")
            
            
            total_number = total_number+number1
            correct_number=correct_number+correct1
            cnt_dict_showup = cnt_dict_showup + cnt_dict_show1
            cnt_dict_correct = cnt_dict_correct + cnt_dict_correct1
            cnt_dict_gt = cnt_dict_gt + cnt_dict_gt1
            ap = np.divide(cnt_dict_correct, cnt_dict_showup)
            ar = np.divide(cnt_dict_correct, cnt_dict_gt)
            mAP= np.nanmean(ap)    
            mAR= np.nanmean(ar)
            
            weighted1=0.3
            weighted2=0.3
            weighted3=0.4
            weighted4=1.0
            weighted5=0.4
            #loss_all = loss1 + loss2 + loss3
            loss_all = loss1*weighted1 + loss2*weighted2 + (loss5)*weighted5
            print ("Epoch:",eidx," No:", iter_cnt, ", Loss all: ", loss_all, \
                   "loss1: ", loss1, "loss2: ", loss2, "loss3: ", loss3, "loss4:", loss4, "loss5:", loss5, \
                   "accuracy:", correct_number/total_number, "mAP: ", mAP, "mAR: ", mAR)
            #scheduler.step()

            #loss_all=loss_all / back_batch_time

            params_new=params_gen(charnet)
            #paramsdiff=(params_new[0]['params']-params[0]['params'])
            #print('Params diff sum after backward()', paramsdiff)
            batch_times = batch_times + 1
        
            if batch_times >= back_batch_time:

                batch_times = 0
        

        
            iter_cnt = iter_cnt + 1
            time = time + 1
            if debug==True and time == 10:
                exit()




            state = {
                    'state_dict' : charnet.state_dict(),
 
                    }
            

            images=None
            images_np=None
            pred_word_orient=None
            pred_word_fg=None
            pred_word_tblr=None
            pred_char_fg=None
            pred_char_tblr=None
            pred_char_orient=None
            pred_char_cls=None
            char_bboxes=None
            char_scores=None 
            polygon_chars=None
            line_chars=None
            ss_word_bboxes=None
            indexes=None
            
            if (debug==True):print("Memory usage", psutil.Process().memory_info().rss / (1024 * 1024))
            
            
            if debug:
                snapshotN = tracemalloc.take_snapshot()
                snapshotN.filter_traces((tracemalloc.Filter(True, "loss"),
                                     tracemalloc.Filter(True, "<unknown>"),))
                top_stats = snapshotN.statistics('lineno')
                #top_stats = snapshotN.compare_to(snapshotO, 'lineno')
                snapshotO = snapshotN
                print("[Top 10]")
                for stat in top_stats[:10]:
                    print(stat)
            

            if (debug==True):
                print("GPU Usage after emptying the cache")
                gpu_usage()
                
    
    
    
    params=params_gen(charnet)
    
    return params, ap, ar, mAP, mAR








def test_model2(args, cfg):
    
    charnet = CharNet()
    #charnet = CharNet(backbone=InternImage(channels=256, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32],layer_scale=1.0))
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
    
    main_cfg=Trainsetting_conf['Experiment']['main']
    train_tag=main_cfg['train']
    valid_tag=main_cfg['valid']
    
    train_cfg=Trainsetting_conf['Experiment']['train']
    train_cfg.update(cmd=cmd_in)
 
    #train_synth_cfg=Trainsetting_conf['Experiment']['train_synth']
    #train_synth_cfg=Trainsetting_conf['Experiment']['train_basket']
    #train_synth_cfg=Trainsetting_conf['Experiment']['valid_basket']
    #train_synth_cfg=Trainsetting_conf['Experiment']['valid_game']
    train_synth_cfg=Trainsetting_conf['Experiment'][valid_tag]

    train_synth_cfg.update(cmd=cmd_in)
    
    train_img_loader = Configurable.construct_class_from_config(train_cfg)
    train_synth_img_loader = Configurable.construct_class_from_config(train_synth_cfg)
    
    
    
    myprocess=train_synth_img_loader.data_loader.dataset.processes;
    #mydataset=train_synth_img_loader.data_loader.dataset
    #myprocess=train_synth_img_loader.data_loader.dataset[22]
    data = {}
    
    #target = train_synth_img_loader.data_loader.dataset.targets[1]
    #target_char = train_synth_img_loader.data_loader.dataset.targets_char[1]
    #image_path=train_synth_img_loader.data_loader.dataset.image_paths[973]
    #data=train_synth_img_loader.data_loader.dataset.getData(973)
    
    #image_path=train_synth_img_loader.data_loader.dataset.image_paths[679]
    #data=train_synth_img_loader.data_loader.dataset.getData(679)
    #data['index']=679
    
    image_path=train_synth_img_loader.data_loader.dataset.image_paths[172]
    data=train_synth_img_loader.data_loader.dataset.getData(172)
    data['index']=172
    
    #image_path=train_synth_img_loader.data_loader.dataset.image_paths[444]
    #data=train_synth_img_loader.data_loader.dataset.getData(444)
    #data['index']=444
    
    imgr = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    ##imgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    data['filename']=image_path
    data['data_id']=image_path
    data['image']=imgr

    #train_synth_img_loader.data_loader.dataset.is_training=True

    
    #data5=train_synth_img_loader.data_loader.dataset[1]
    #data['lines']=target
    #data['chars']=target_char
    #img_pathd="./test.jpg"
    #imgd=cv2.imread(img_pathd)
    
    data1=myprocess[0](data)
    data2=myprocess[1](data1)
    data3=myprocess[2](data2)
    data4=myprocess[3](data3)
    data5=myprocess[4](data4)
    #data6=myprocess[5](data5)

    data_oneT=train_synth_img_loader.data_loader.dataset[1991]
    debug1=False
    if debug1:
        #print("Image Path", data_oneT['filename'])
        invTrans=train_synth_img_loader.data_loader.dataset.processes[4]
        img=invTrans.lib_inv_trans(data_oneT['image'])
        blend_img=blending_two_imgs(img, data_oneT['score_map'].astype('uint8'))
        #blend_char_img=blending_two_imgs(img, score_map_char[0].cpu().numpy().astype('uint8'))
        #score_map_char_mask=torch.where(score_map_char> 0, 1, 0)
        blend_char_img=blending_two_imgs(img, data_oneT['score_map_char'].astype('uint8'))
        #draw_polys(blend_img, data_oneT['polygons'])
        draw_polys(blend_char_img, data_oneT['polygons_char'])
        cv2.destroyAllWindows()
        cv2.imshow("score_map", blend_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imshow("score_map_char", blend_char_img)
        cv2.waitKey()




    #dataf=mydataset[0]
    
    #dlen = len(train_synth_img_loader.data_loader)
    #print ('Len = ', dlen)
    
    #for i in range(dlen):
    #    temp=train_synth_img_loader.data_loader.dataset[i]
    #    print('image len =', len(temp['image']))
    
    
    
    
    
#    score_map, geo_map, training_mask, rects=generate_rbox((data6['image'].shape[1], data6['image'].shape[0]), data6['polygons'], data6['lines_text'])
    #data7 = myconv(data6['image'])
    #data7=myprocess[6](data6)
    
    #0 <data.processes.augment_data.AugmentDetectionData >,    -> affine transforer, Flip, resize
    #1 <data.processes.random_crop_data.RandomCropData >,      -> Random Crop Data
    #2 <data.processes.make_icdar_data.MakeICDARData >,        -> Bonding box transfer to t,b,l,r, angle
    #3 <data.processes.make_seg_detection_data.MakeSegDetectionData >, remove 
    #4->3 <data.processes.make_border_map.MakeBorderMap >,     -> Shrink Bonding box and bonding box generation 
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
    
    #charnet = CharNet(hourglass88GCN())
    charnet = CharNet()
    #charnet = CharNet(backbone=InternImage(channels=256, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32],layer_scale=1.0))
    if(cfg.WEIGHT !=''):
        charnet.load_state_dict(torch.load(cfg.WEIGHT))
    #Trainsetting(Trainsetting_conf['Experiment']['train']])
#Train without profile
    #train_model(charnet, args, cfg, train_synth_img_loader.data_loader, train_synth_cfg)
    
    debug_parameters=True
    if debug_parameters:
        tcnt=0
        for name, param in charnet.named_parameters():
            if param.requires_grad:
                print("name = ", name,", size of = ", param.data.size(), "product = ", np.product(param.data.size()))
                tcnt=tcnt+np.product(param.data.size())
    
    print ("Parameters count = ", tcnt)
    
    params, ap50, ar50, mAP50, mAR50=validate_model(charnet, args, cfg, train_synth_img_loader.data_loader, train_synth_cfg, 0.5)
    params, ap75, ar75, mAP75, mAR75=validate_model(charnet, args, cfg, train_synth_img_loader.data_loader, train_synth_cfg, 0.75)
    
    ap=0.0
    ar=0.0
    mAP=0.0
    mAR=0.0
    
    for idx in range(0, 10):
        params, aptmp, artmp, mAPtmp, mARtmp=validate_model(charnet, args, cfg, train_synth_img_loader.data_loader, train_synth_cfg, 0.5+0.05*idx)
        ap=ap+aptmp
        ar=ar+artmp
        mAP=mAP+mAPtmp
        mAR=mAR+mARtmp
        

    print ("mAP50: ", mAP50, "mAR50: ", mAR50)
    print ("mAP75: ", mAP75, "mAR75: ", mAR75)
    print ("mAP: ", mAP/10, "mAR: ", mAR/10)
    print ("Parameters count = ", tcnt)

#Train code with profiling
    #cProfile.run('train_model(args, cfg, train_synth_img_loader.data_loader)', 'restats')
    #p = pstats.Stats('restats')
    #p.sort_stats(SortKey.TIME).print_stats(50)
    #p.sort_stats(SortKey.CUMULATIVE).print_stats(50)
    

#TEST code
    #test_model(args, cfg)
