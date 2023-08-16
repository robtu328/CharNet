# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from interimage.ops_dcnv3 import modules as opsm
#from interimage.intern_image import StemLayer

#from interimage.intern_image import InternImage

_norm_func = lambda num_features: nn.BatchNorm2d(num_features, eps=1e-5)


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN',
                 order='channels_last'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      order)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


def _make_layer(in_channels, out_channels, num_blocks, **kwargs):
    blocks = []
    blocks.append(Residual(in_channels, out_channels))
    for _ in range(1, num_blocks):
        blocks.append(Residual(out_channels, out_channels, **kwargs))
    return nn.Sequential(*blocks)


def _make_layer_revr(in_channels, out_channels, num_blocks, **kwargs):
    blocks = []
    for _ in range(num_blocks - 1):
        blocks.append(Residual(in_channels, in_channels, **kwargs))
    blocks.append(Residual(in_channels, out_channels, **kwargs))
    return nn.Sequential(*blocks)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            _norm_func(out_channels),
            nn.ReLU()
        )
        
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            _norm_func(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                _norm_func(out_channels)
            )
        else:
            self.skip = None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        b1 = self.conv_2(self.conv_1(x))
        if self.skip is None:
            return self.out_relu(b1 + x)
        else:
            return self.out_relu(b1 + self.skip(x))


class HourGlassBlock(nn.Module):
    def __init__(self, n, channels, blocks):
        super(HourGlassBlock, self).__init__()

        self.up_1 = _make_layer(channels[0], channels[0], blocks[0])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low_1 = _make_layer(channels[0], channels[1], blocks[0])
        if n <= 1:
            self.low_2 = _make_layer(channels[1], channels[1], blocks[1])
        else:
            self.low_2 = HourGlassBlock(n - 1, channels[1:], blocks[1:])
        self.low_3 = _make_layer_revr(channels[1], channels[0], blocks[0])

    def forward(self, x):
        upsample = lambda input: F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        up_1 = self.up_1(x)
        low = self.low_3(self.low_2(self.low_1(self.pool(x))))
        return upsample(low) + up_1


class HourGlassNet(nn.Module):
    def __init__(self, n, channels, blocks):
        super(HourGlassNet, self).__init__()
        
        #core_op=getattr(opsm, 'DCNv3_pytorch')
        
        
        self.pre = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False), #channels = 128
            _norm_func(128),
            nn.ReLU(),
            Residual(128, 256, stride=2)    #channels in= 128, channels out=256
            #nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3, bias=False), #channels = 128
            #_norm_func(256),
            #nn.ReLU(),
            #Residual(256, 512, stride=2)    #channels in= 128, channels out=256            
        )
        hourglass_blocks = []
        for _ in range(2):
            hourglass_blocks.append(
                HourGlassBlock(n, channels, blocks)
            )
        self.hourglass_blocks = nn.Sequential(*hourglass_blocks)

    def forward(self, x):
        return self.hourglass_blocks(self.pre(x))


class HourGlassNetGCN(nn.Module):
    def __init__(self, n, channels, blocks):
        super(HourGlassNetGCN, self).__init__()
        
        core_op=getattr(opsm, 'DCNv3_pytorch')
        
        
        self.pre = nn.Sequential(
            StemLayer(in_chans=3, out_chans=128, act_layer='GELU', norm_layer='LN', order='channels_first'),
            core_op( channels=128, kernel_size=3 , stride=1, pad=1,dilation=1,
                     group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN',
                     dw_kernel_size=None, center_feature_scale=False, imgFmt='CHW'),
            Residual(128, 256, stride=1)    #channels in= 128, channels out=256
        )
        
        #self.pre = nn.Sequential(
        #    nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False), #channels = 128
        #    _norm_func(128),
        #    nn.ReLU(),
        #    Residual(128, 256, stride=2)    #channels in= 128, channels out=256
        #)
        hourglass_blocks = []
        for _ in range(2):
            hourglass_blocks.append(
                HourGlassBlock(n, channels, blocks)
            )
        self.hourglass_blocks = nn.Sequential(*hourglass_blocks)

    def forward(self, x):
        return self.hourglass_blocks(self.pre(x))





def hourglass88():
    return HourGlassNet(3, [256, 256, 256, 512], [2, 2, 2, 2])

def hourglass88v1():
    return HourGlassNet(3, [256, 512, 512, 512], [2, 2, 2, 2])
    #return HourGlassNet(3, [256, 256, 256, 512], [2, 2, 2, 2])


def hourglass88GCN():
    return HourGlassNetGCN(3, [256, 256, 256, 512], [2, 2, 2, 2])
