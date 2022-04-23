import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import scipy.io


class SynthDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    mat_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, mat_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.mat_list = mat_list or self.mat_list
        
        if 'train' in self.mat_list[0]:
            self.is_training = True
        else:
            self.is_training = False
            
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_maps = []
        self.gt_maps_char = []
        
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            print("Load Mat file ", self.data_dir)
            mat = scipy.io.loadmat(self.mat_list[i])
            print("Done...")
            
            gt_map = []
            gt_map_char = []
            #with open(self.mat_list[i], 'r') as fid:
                
            image_list = mat['imnames'][0]
            gt_word_list = mat['wordBB'][0]
            gt_char_list = mat['charBB'][0]
            txt_list = mat['txt'][0]
            
            if self.is_training:
                image_path = [self.data_dir[i]+timg[0] for timg in image_list]
                for timg, ttxt in zip(gt_word_list, txt_list):
                   if(len(timg.shape)==2):
                     timg=timg.reshape(-1, 4, 1)  
                   gt_map.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
                gt_map_char= [''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split()) for ttxt in txt_list]
            else:
                image_path = [self.data_dir[i]+timg[0] for timg in image_list]
                for timg, ttxt in zip(gt_word_list, txt_list):
                   if(len(timg.shape)==2):
                     timg=timg.reshape(-1, 4, 1)  
                   gt_map.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
                
                for timg, ttxt in zip(gt_char_list, txt_list):
                   if(len(timg.shape)==2):
                     timg=timg.reshape(-1, 4, 1)  
                   ttxt=''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split())
                   gt_map_char.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
                
                
                
   #             gt_map_char= [''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split()) for ttxt in txt_list]

           # tmp=np.reshape(mat['txt'][0][0], (1, -1))
#''.join((''.join(np.reshape(mat['txt'][0][0], (1, -1)).tolist()[0])).split())
#''.join(tmp1.split())


            self.image_paths += image_path
            self.gt_maps += gt_map
            self.gt_maps_char += gt_map_char
            #print(image_path, " ", gt_path)
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        self.targets_char = self.load_ann_char()
        
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_maps:
            lines = []
            
            for line, text in zip(gt['poly'], gt['txt']):
                item = {}

                item['poly'] = line.round().tolist()
                item['text'] = text
                lines.append(item)
            res.append(lines)
        return res

    def load_ann_char(self):
        res = []
        for gt in self.gt_maps_char:
            lines = []
            
            for line, text in zip(gt['poly'], gt['txt']):
                item = {}

                item['poly'] = line.round().tolist()
                item['text'] = text
                lines.append(item)
            res.append(lines)
        return res


    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
