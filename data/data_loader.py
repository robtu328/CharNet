import math
import bisect

import imgaug
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, ConcatDataset, BatchSampler

from concern.config import Configurable, State


import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_fn(batch):
    #img, score_map, geo_map, training_mask = zip(*batch)#tuple
    #image, polygon, polygon_char, lines_text, lines_char, gt, mask, gt_char, mask_char, thresh_map, thresh_mask, thresh_map_char, thresh_mask_char = zip(*batch)
    bs = len(batch)
    #print('batch type', type(batch), 'Length =', bs)

    #print("Length = ", bs, "thresh_map = ", thresh_map)
    images = []
    images_np = []
#    polygons = []
#    polygon_chars = []
    lines_texts = []
    lines_chars = []
#    gts = []
#    masks = []
#    gt_chars = []
#    mask_chars = []
#    thresh_maps = []
#    thresh_masks = []
#    thresh_map_chars = []
#    thresh_mask_chars = []
    score_map = []
    geo_map = []
    training_mask = []
    score_map_char = []
    geo_map_char = []
    training_mask_char = [] 
    polygon_chars = []
    indexes = []
    
   
    for index in range(bs):
        #print ('index = ', index)
        if batch[index]['image'] is not None:
            #a = torch.from_numpy(batch[index]['image'])
            #a = a.permute(2, 1, 0)
            a = batch[index]['image']
            images.append(a)
            images_np.append(batch[index]['image'])
           
            #b = torch.FloatTensor(batch[index]['polygons'])
            #b = b.permute(2, 0, 1)
#            polygons.append(batch[index]['polygons'])
            
            #c = torch.FloatTensor(batch[index]['polygons_char'])
            #c = c.permute(2, 0, 1)
            polygon_chars.append(batch[index]['polygons_char'])
            
            #d = torch.from_numpy(batch[i]['lines_text'])
            lines_texts.append(batch[index]['lines_text'])

            #e = torch.from_numpy(batch[i]['lines_char'])
            lines_chars.append(batch[index]['lines_char'])
            
            f = torch.from_numpy(batch[index]['score_map'])
            score_map.append(f[::4, ::4])
            
            g = torch.from_numpy(batch[index]['geo_map']/4)
            geo_map.append(g[::4, ::4])
            
            h = torch.from_numpy(batch[index]['training_mask'])
            training_mask.append(h[::4, ::4])
            
            i = torch.from_numpy(batch[index]['score_map_char'])
            score_map_char.append(i[::4, ::4])
            
            j = torch.from_numpy(batch[index]['geo_map_char']/4)
            geo_map_char.append(j[::4, ::4])
            
            k = torch.from_numpy(batch[index]['training_mask_char'])
            training_mask_char.append(k[::4, ::4])    
            
            indexes.append(batch[index]['index'])

#           f = torch.from_numpy(batch[index]['gt'])
#            gts.append(f)

#           g = torch.from_numpy(batch[index]['mask'])
#            masks.append(g)

#            h = torch.from_numpy(batch[index]['gt_char'])
#            gt_chars.append(h)

#            i = torch.from_numpy(batch[index]['mask_char'])
#            mask_chars.append(i)
            
            
#            j = torch.from_numpy(batch[index]['thresh_map'])
#            thresh_maps.append(j)    
    
    
#            k = torch.from_numpy(batch[index]['thresh_mask'])
#            thresh_masks.append(k)    

#            l = torch.from_numpy(batch[index]['thresh_map_char'])
#            thresh_map_chars.append(l)
    
#            m = torch.from_numpy(batch[index]['thresh_mask_char'])
#            thresh_mask_chars.append(m)

    images = torch.stack(images, 0)
    score_map = torch.stack(score_map, 0)
    geo_map = torch.stack(geo_map, 0)
    training_mask = torch.stack(training_mask, 0)
    score_map_char = torch.stack(score_map_char, 0)
    geo_map_char = [torch.stack(geo_map_char, 0)]
    training_mask_char = torch.stack(training_mask_char, 0)    
    #polygons = torch.stack(polygons, 0)
    #polygon_chars = torch.stack(polygon_chars, 0)
    #lines_texts = torch.stack(lines_texts, 0)
    #lines_chars = torch.stack(lines_chars, 0)
#    gts = torch.stack(gts, 0)
#    masks = torch.stack(masks, 0)
#    gt_chars = torch.stack(gt_chars, 0)
#    mask_chars = torch.stack(mask_chars, 0)
#    thresh_maps = torch.stack(thresh_maps, 0)
#    thresh_masks = torch.stack(thresh_masks, 0)
#    thresh_map_chars = torch.stack(thresh_map_chars, 0)
#    thresh_mask_chars = torch.stack(thresh_mask_chars, 0)
    

    #return images, score_maps, geo_maps, training_masks
    #return images, polygons, polygon_chars, lines_texts, lines_chars, gts, masks, gt_chars, mask_chars, thresh_maps, thresh_masks, thresh_map_chars, thresh_mask_chars 
    return images, score_map, geo_map, training_mask, score_map_char, geo_map_char, training_mask_char, images_np,  polygon_chars, lines_chars, indexes

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    #print ("Batch Type = ", type(batch), "Len ", len(batch))#, "index 0 key", batch[0].keys())
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        #print ("Batch TypeII = ", type(batch), "Len ", len(batch))#, "index 0 key", batch[0])

        it = iter(batch)
        elem_size = len(next(it))
        #if not all(len(elem) == elem_size for elem in it):
            #for elem in it:   
            #    print (elem)
            #    print (len(elem), "!=", elem_size, " ")
            #raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))





def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    imgaug.seed(worker_id)


class TrainSettings(Configurable):
    data_loader = State()
    epochs = State(default=10)

    def __init__(self, **kwargs):
        #kwargs['cmd'].update(is_train=True)
        self.load_all(**kwargs)
        if 'epochs' in kwargs['cmd']:
            self.epochs = kwargs['cmd']['epochs']


class DataLoader(Configurable, torch.utils.data.DataLoader):
    dataset = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    is_train = State(default=True)
    collect_fn = State(default=None)
    drop_last = State(default=True)
    shuffle = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)
        if self.collect_fn is None:
            #self.collect_fn = torch.utils.data.dataloader.default_collate
            self.collect_fn = collate_fn
        cmd = kwargs.get('cmd', {})
        self.is_train = cmd['is_train']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']
        if self.shuffle is None:
            self.shuffle = self.is_train
        self.num_workers = cmd.get('num_workers', self.num_workers)

        if cmd.get('distributed'):
            sampler = DistributedSampler(
                self.dataset, shuffle=self.shuffle,
                num_replicas=cmd['num_gpus'])
            batch_sampler = BatchSampler(
                sampler, self.batch_size//cmd['num_gpus'], False)
            torch.utils.data.DataLoader.__init__(
                self, self.dataset, batch_sampler=batch_sampler,
                num_workers=self.num_workers, pin_memory=False,
                drop_last=self.drop_last, collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        else:
            torch.utils.data.DataLoader.__init__(
                self, self.dataset,
                batch_size=self.batch_size, num_workers=self.num_workers,
                drop_last=self.drop_last, shuffle=self.shuffle,
                pin_memory=True, collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        self.collect_fn = str(self.collect_fn)



class SuccessiveRandomSampler(Sampler):
    '''Random Sampler that yields sorted data in successive ranges.
    Args:
        dataset: Dataset used for sampling.
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteOrderedSampler(Sampler):
    def __init__(self, data_source, limit_size):
        self.data_source = data_source
        self.limit_size = limit_size

    def __iter__(self):
        n = len(self.data_source)

        def wrapper():
            cnt = 0
            while cnt < self.limit_size:
                if cnt % n == 0:
                    idx = torch.randperm(n).tolist()
                yield idx[cnt % n]
                cnt += 1
        return wrapper()

    def __len__(self):
        return self.limit_size


class InfiniteDataLoader(Configurable, torch.utils.data.DataLoader):
    dataset = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    limit_size = State(default=2 ** 31)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']

        sampler = InfiniteOrderedSampler(self.dataset, self.limit_size)

        torch.utils.data.DataLoader.__init__(
            self, self.dataset,
            batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, worker_init_fn=default_worker_init_fn,
        )


class RandomSampleSampler(Sampler):
    def __init__(self, data_source, weights=None, size=2 ** 31):
        self.data_source = data_source
        if weights is None:
            self.probabilities = np.full(len(data_source), 1 / len(data_source))
        else:
            self.probabilities = np.array(weights) / np.sum(weights)
        self.cum_prob = np.cumsum(self.probabilities)
        self.size = size

    def __iter__(self):
        def wrapper():
            for i in range(self.size):
                yield bisect.bisect(self.cum_prob, torch.rand(1)[0], hi=len(self.data_source) - 1)
        return wrapper()

    def __len__(self):
        return self.size


class RandomSampleDataLoader(Configurable, torch.utils.data.DataLoader):
    datasets = State()
    weights = State()
    batch_size = State(default=256)
    num_workers = State(default=10)
    size = State(default=2 ** 31)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']

        probs = []
        for dataset, weight in zip(self.datasets, self.weights):
            probs.append(np.full(len(dataset), weight / len(dataset)))

        dataset = ConcatDataset(self.datasets)
        probs = np.concatenate(probs)
        assert(len(dataset) == len(probs))

        sampler = RandomSampleSampler(dataset, probs, self.size)

        torch.utils.data.DataLoader.__init__(
            self, dataset,
            batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, worker_init_fn=default_worker_init_fn,
        )
