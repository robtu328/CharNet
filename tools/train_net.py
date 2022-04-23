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
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt
from data.image_dataset import ImageDataset
from data.synth_dataset import SynthDataset
from data.data_loader import DataLoader, TrainSettings

from concern.config import Configurable, Config

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
    charnet.cuda()
    params=params_gen(charnet)
    
    optimizer = torch.optim.SGD(params, momentum=0.001)
    
    
    for batch in img_loader:
        for im in range(len(batch)):
            char_bboxes, char_scores, word_instances = charnet(im['image'], 640, 640, 640, 640)
        
        
        #Check  Image size. and transofrmation reuslt
#        print("Processing {}...".format(im_name))
#        im_file = os.path.join(args.image_dir, im_name)
#        im_original = cv2.imread(im_file)
#        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
#        with torch.no_grad():
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
    
    
    
    myprocess=train_synth_img_loader.dataset.processes
    data = {}
    image_path=train_synth_img_loader.dataset.image_paths[1]
    target = train_synth_img_loader.dataset.targets[1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    data['filename']=image_path
    data['data_id']=image_path
    data['image']=img
    data['lines']=target
    data1=myprocess[0](data)
    data2=myprocess[1](data1)
    data3=myprocess[2](data2)
    data4=myprocess[3](data3)
    data5=myprocess[4](data4)
    data6=myprocess[5](data5)

    
    
    
    
    
    
    
    #Trainsetting(Trainsetting_conf['Experiment']['train']])

#Train code
    train_model(args, cfg, train_img_loader)

#TEST code
    test_model(args, cfg)
