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
#import data.data_utils
from data.data_utils import generate_rbox

def preprocess_words(word_ar):
    words = []
    for ii in range(np.shape(word_ar)[0]):
        s = word_ar[ii]
        start = 0
        while s[start] == ' ' or s[start] == '\n':
            start += 1
        for i in range(start + 1, len(s) + 1):
            if i == len(s) or s[i] == '\n' or s[i] == ' ':
                if start != i:
                    words.append(s[start : i])
                start = i + 1
    return words

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
                   ttxt=preprocess_words(ttxt)
                   gt_map.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
                   
                for timg, ttxt in zip(gt_char_list, txt_list):
                   if(len(timg.shape)==2):
                     timg=timg.reshape(-1, 4, 1)  
                   ttxt=''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split())
                   gt_map_char.append({'poly':np.transpose(timg, (2,1,0)),'txt':ttxt})
                   
                   
#                gt_map_char= [''.join((''.join(np.reshape(ttxt, (1, -1)).tolist()[0])).split()) for ttxt in txt_list]
            else:
                image_path = [self.data_dir[i]+timg[0] for timg in image_list]
                for timg, ttxt in zip(gt_word_list, txt_list):
                   if(len(timg.shape)==2):
                     timg=timg.reshape(-1, 4, 1)  
                   ttxt=preprocess_words(ttxt)
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
        #for i in range(len(self.gt_maps_char)):
            #gt = self.gt_maps_char[i]
            lines = []
            if (len(gt['txt']) != gt['poly'].shape[0]):
                raise NameError('Charter box size do not match txt lenth', len(gt['txt']), 'to ', gt['poly'].shape[0])
            
            if '##' in gt['txt']:
                raise NameError('# is found string = ', gt['txt'])
                
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
        target_char = self.targets_char[index]
        data['lines'] = target
        data['chars'] = target_char
        #print ("image size=", img.shape)
        #print("Start index process ", index)
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
            
            #score_map, geo_map, training_mask, rects=generate_rbox((data['image'].shape[1], data['image'].shape[0]), data['polygons'], data['lines_text'])
            #data['score_map']=score_map
            #data['geo_map']=geo_map
            #data['training_mask']=training_mask
            #data['rects']=rects
        #print("End index process, image size =", data['image'].shape, "  index = ", index)
        #print("End index process, len(data) =", len(data))
        
        #score_map, geo_map, training_mask = generate_rbox((img.shape[1], img.shape[0]), data['lines']['poly'], data['lines']['text'])
        
        #print (data.keys())
        return data

    def __len__(self):
        return len(self.image_paths)
        #return len(self.image_paths)