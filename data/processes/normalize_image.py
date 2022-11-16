import numpy as np
import torch
import warnings
from concern.config import State

from .data_process import DataProcess
from torchvision import transforms

class NormalizeImage(DataProcess):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    norm_type = State(default="lib")
 
    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.debug = False
        warnings.simplefilter("ignore")
        to_rgb_transform = transforms.Lambda(lambda x: x[[2, 1, 0]]) #RGB to BGR
        
        self.normalize = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
        
        self.inv_normal = transforms.Compose([ 
           transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), 
           transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]), 
           to_rgb_transform
           ])
        self.cnormalize = self.build_transform()

    def build_transform(self):
        to_rgb_transform = transforms.Lambda(lambda x: x[[2, 1, 0]]) #BGR -> RGB

        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose(
            [
               # transforms.ToPILImage(),
                transforms.ToTensor(),    # HWC -> CHW
                to_rgb_transform,         # BGR -> RGB     
                normalize_transform,
            ]
        )
        return transform        
            
    def lib_trans(self, image):
        
        return self.cnormalize(image)
        #return self.normalize(torch.from_numpy(image).permute(2, 1, 0)).float()
        
    def lib_inv_trans(self, image):
        
        image1=(self.inv_normal(image.to('cpu')).numpy()*255).astype('uint8') 
        
        return np.ascontiguousarray(np.transpose(image1, (1,2,0)))
        
    def manual_trans(self, image):
        
        image -= self.RGB_MEAN
        image /= 255.
        #image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.from_numpy(image).permute(2, 1, 0).float()
        
        return image
    
    def manual_inv_trans(self, image):
        #image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image.permute(2, 1, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)       
        
        return image

    def process(self, data):
        
        assert 'image' in data, '`image` in data is required by this process'
        image = data['image']
        
        if(self.norm_type == 'lib'):
            
            data['image'] = self.lib_trans(image.astype('uint8'))
        else:
            data['image'] = self.manual_trans(image.astype('uint8'))
        
        return data
        #assert 'image' in data, '`image` in data is required by this process'
        #image = data['image']
        #image -= self.RGB_MEAN
        #image /= 255.
        #image = torch.from_numpy(image).permute(2, 0, 1).float()
        #image = torch.from_numpy(image).permute(2, 1, 0).float()
        #data['image'] = image
        #return data

    @classmethod
    def restore(self, image):
        #image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image.permute(2, 1, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image
