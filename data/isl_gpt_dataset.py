# -*- coding: utf-8 -*-
"""
Another version of Silicio Datset
"""
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os



class ISLGPTDataset(Dataset):
    condition = {
        'A': ['Rubin/scott_1_0'],
        'B': [f'Finkbeiner/kevan_0_{i}' for i in [7, 8, 9, 10]],
        'C': ['Finkbeiner/alicia_2_0'],
        'D': ['Finkbeiner/yusha_0_1'],
        'E': ['GLS/2015_06_26']
    }

    gt_info = {
        'DAPI_CONFOCAL': 0,
        'CELLMASK_CONFOCAL': 1,
        'NFH_CONFOCAL': 2,
        'MAP2_CONFOCAL': 3,
        'TUJ1_WIDEFIELD': 4,
        'ISLET_WIDEFIELD': 5,
        'DAPI_WIDEFIELD': 6,
        'DEAD_CONFOCAL': 7
    }

    def __init__(self, args, phase=None, random_seed=0):
        self.random_seed = random_seed
        random.seed(random_seed)
        self.args = args.DATA
        self.phase = phase
        self.data_root = args.DATA.DATA_PATH
        json_path = args.DATA.TRAIN_JSON_FILE if self.phase == 'train' else args.DATA.TEST_JSON_FILE
        with open(json_path) as f:
            self.data_items = json.load(f)
        # sample data for setting hyper-parameters
        self._condition_query = {i:k for k, v in self.condition.items() for i in v}

    def __len__(self):
        return len(self.data_items)

    def read_img(self, item_info, phase):

        # print(item_info)
        # print(gt_type)
        _prefix = f'patches_{phase}'
        keys = [f'{i}' for i in range(13)]
        coors = [str(i) for i in item_info['coors']]
        # read input images
        input_names = [item_info['inputs'][k]+','+'_'.join(coors)+'.png' for k in keys]
        input_images = []
        for name in input_names:
            p = os.path.join(self.data_root, _prefix, item_info['prefix'], name)
            img = Image.open(p)
            input_images.append(img)
        # read gt images
        gt_images = []
        gt_types = []
        gt_names = [(k, v + ','+'_'.join(coors)+'.png') for k, v in item_info['labels'].items() if k != 'NEURITE_CONFOCAL']
        gt_file_names = []
        for k, name in gt_names:
            p = os.path.join(self.data_root, _prefix, item_info['prefix'], name)
            gt_file_names.append(name)
            gt = Image.open(p)
            gt_images.append(gt)
            gt_types.append(k)
        return input_images, gt_images, gt_types, gt_file_names

    def multi_scale_crop(self, input_data):
        input_middle = [functional.center_crop(i, self.args.IMG_SIZE[1]) for i in input_data]
        input_small = [functional.center_crop(i, self.args.IMG_SIZE[0]) for i in input_data]
        input_big = [functional.center_crop(i, self.args.IMG_SIZE[2]) for i in input_data]
        return input_small, input_middle, input_big

    def center_crop(self, img, size, channel_first=True):
        """Tensor center crop
        """
        if channel_first:
            _, h, w = img.shape
            y, x = (h-size)//2, (w-size)//2
            img = img[:, y:y+size, x:x+size]
        else:
            h, w, _ = img.shape
            y, x = (h-size)//2, (w-size)//2
            img = img[y:y+size, x:x+size]
        return img

    def __getitem__(self, item):
        item_info = self.data_items[item]
        # read input images
        input_images, input_gt, gt_types, gt_file_names = self.read_img(item_info, self.phase)
        # crop multi-scale images
        input_small, input_middle, input_big = self.multi_scale_crop(input_images)
        input_gt = [functional.center_crop(i, self.args.IMG_SIZE[1]) for i in input_gt]

        # random flip
        if self.phase == 'train':
            flip_func = self.flip()
            input_small = flip_func(input_small)
            input_middle = flip_func(input_middle)
            input_big = flip_func(input_big)
            input_gt = flip_func(input_gt)

        input_gt = self.normalize(input_gt)
        # define mask
        mask = torch.zeros((self.args.MODALITY_NUM, self.args.IMG_SIZE[1], self.args.IMG_SIZE[1]), dtype=torch.float32)
        gt = torch.zeros_like(mask)
        gt_index = []
        for tmp_gt, gt_type in zip(input_gt, gt_types):
            index = self.gt_info[gt_type]
            gt_index.append(index)
            mask[index]= 1.0
            gt[index] = tmp_gt
            
        input_data = [input_small, input_middle, input_big]
        input_data = [torch.cat(self.normalize(i), dim=0) for i in input_data]
        # Log transform
        if self.args.LOG_TRANSFORM:
            input_data = [torch.log2(i+1) for i in input_data]
                    
        
        data = {
            'mask': mask,
            'input': input_data,
            'gt': gt,
            'input_type': item_info['input_type'],
            'gt_type': gt_types,
            'gt_index': gt_index,
            'prefix': item_info['prefix'],
            'condition': self._condition_query[item_info['prefix']],
            'gt_names': gt_file_names
        }
        return data

    def __repr__(self):
        class_info = 'Class: {}\n'.format(type(self).__name__)
        phase = 'Phase: {} data\n'.format(self.phase)
        total_patches = 'Total patches: {}\n'.format(len(self))
        data_info = {}
        for item in self.data_items:
            prefix_dict = data_info.setdefault(item['prefix'], {})
            for k, _ in item['labels'].items():
                num = data_info[item['prefix']].setdefault(k, 0)
                data_info[item['prefix']][k] = num + 1
        data_info = ''.join([f'\t>>{k}:{v}\n' for k, v in data_info.items()])
        # data_info = f'{data_info}'
        out_put = ''.join([
            class_info, phase, total_patches, data_info
        ])
        return out_put


    @staticmethod
    def normalize(imgs: list):
        results = []
        v = 2 ** 16
        for img in imgs:
            img = functional.to_tensor(img)
            # u16 -> (0, 1)
            results.append(img.float()/v)
        return results

    @staticmethod
    def flip():
        h = random.random()
        v = random.random()        
        def func(input_data):
            input_data = [functional.hflip(i) for i in input_data] if h > 0.5 else input_data
            input_data = [functional.vflip(i) for i in input_data] if v > 0.5 else input_data
            return input_data
        return func


def collate_fn(batch):
    data = {}
    tensor_keys = ['mask', 'input', 'gt']
    keys = batch[0].keys()
    get_data = lambda x, i: [y[i] for y in x]
    for k in keys:
        tmp = []
        for i in range(len(batch)):
            d = batch[i][k]
            tmp.append(d)
        if k in tensor_keys:
            if k == 'input':
                tmp = [torch.stack(get_data(tmp, i), dim=0) for i in range(3)]
            else:
                tmp = torch.stack(tmp, dim=0)
        data[k] = tmp
    return data

