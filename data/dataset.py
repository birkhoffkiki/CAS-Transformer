# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/21 10:33
@Auth ： majiabo, jmabq@connect.ust.hk
@File ：dataset.py
@IDE ：PyCharm
dataset implementation
"""
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os
import numpy as np


class SilicoDataset(Dataset):
    condition = {
        'A': ['Rubin/scott_1_0'],
        'B': [f'Finkbeiner/kevan_0_{i}' for i in [7, 8, 9, 10]],
        'C': ['Finkbeiner/alicia_2_0'],
        'D': ['Finkbeiner/yusha_0_1'],
        'E': ['GLS/2015_06_26']
    }

    def __init__(self, args, phase=None, random_seed=0, use_single_label=False):
        self.random_seed = random_seed
        random.seed(random_seed)
        args = args.DATA
        self.args = args
        self.phase = phase
        self.data_root = args.DATA_PATH
        json_path = args.TRAIN_JSON_FILE if self.phase == 'train' else args.TEST_JSON_FILE
        with open(json_path) as f:
            self.data_items = json.load(f)
        self.json_path = json_path
        # split multi-style labels 
        self.paired_single_items = []
        for i, item in enumerate(self.data_items):
            for k, _ in item['labels'].items():
                if k != 'NEURITE_CONFOCAL':
                    self.paired_single_items.append((i, item['prefix'], item['input_type'], k))
                    if use_single_label:
                        break
        # sample data for setting hyper-parameters
        _num = int(len(self.paired_single_items)*self.args.DATA_SAMPLE_RATIO)
        self.paired_single_items = self.paired_single_items[:_num]
        self._condition_query = {i:k for k, v in self.condition.items() for i in v}

    def __len__(self):
        return len(self.paired_single_items)

    def read_img(self, item_info, gt_type, phase):
        def func(gt_type_):
            coors = [str(i) for i in item_info['coors']]
            gt_name = item_info['labels'][gt_type_]+','+'_'.join(coors)+'.png'
            p = os.path.join(self.data_root, f'patches_{phase}', item_info['prefix'], gt_name)
            img = Image.open(p)
            return img

        # print(item_info)
        # print(gt_type)
        _prefix = f'patches_{phase}'
        keys = [f'{i}' for i in range(13)]
        coors = [str(i) for i in item_info['coors']]
        input_names = [item_info['inputs'][k]+','+'_'.join(coors)+'.png' for k in keys]
        gt_name = item_info['labels'][gt_type]+','+'_'.join(coors)+'.png'
        input_images = []
        for name in input_names:
            p = os.path.join(self.data_root, _prefix, item_info['prefix'], name)
            img = Image.open(p)
            input_images.append(img)

        p = os.path.join(self.data_root, _prefix, item_info['prefix'], gt_name)
        gt = Image.open(p)

        baseline_name = item_info['baseline'][gt_type]+','+'_'.join(coors)+'.png'
        pp = os.path.join(self.data_root, _prefix, item_info['prefix'], baseline_name)
        baseline_img = Image.open(pp)
        # return input_images, gt, baseline_img, func
        return input_images, gt, baseline_img, func, gt_name, baseline_name

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
        data_index, prefix, input_type, gt_type = self.paired_single_items[item]
        # read input images
        item_info = self.data_items[data_index]
        input_images, input_gt, input_baseline, read_func, gt_file_name, baseline_name = self.read_img(item_info, gt_type, self.phase)
        # input_images, input_gt, input_baseline, read_func = self.read_img(item_info, gt_type, self.phase)


        # crop multi-scale images
        input_small, input_middle, input_big = self.multi_scale_crop(input_images)
        input_gt = functional.center_crop(input_gt, self.args.IMG_SIZE[1])
        input_baseline = functional.center_crop(input_baseline, self.args.IMG_SIZE[1])

        # random flip
        if self.phase == 'train':
            flip_func = self.flip()
            input_small = flip_func(input_small)
            input_middle = flip_func(input_middle)
            input_big = flip_func(input_big)
            [input_baseline] = flip_func([input_baseline])
            [input_gt] = flip_func([input_gt])

        # define mask
        if random.random() < self.args.FUSION_PROB and self.phase == 'train':
            t_gt = [i for i in self.data_items[data_index]['labels'] if i!=gt_type and i!='NEURITE_CONFOCAL']
            a_gt_type = random.choice(t_gt)
            another_gt = read_func(a_gt_type)
            mode = random.choice(['overwrite', 'mean'])
            mask, input_gt = self.define_mask(input_gt, another_gt, self.args[gt_type], self.args[a_gt_type], mode)
        else:
            mask = torch.zeros((self.args.IMG_SIZE[1], self.args.IMG_SIZE[1], self.args.MODALITY_NUM), dtype=torch.float32)
            mask[..., self.args[gt_type]] = 1.0
            input_gt = torch.from_numpy(np.array(input_gt).astype('float32'))[None]
            
        # apply transformer
        # TODO

        input_data = [input_small, input_middle, input_big]
        input_data = [torch.cat(self.normalize(i), dim=0) for i in input_data]
        input_gt = input_gt/2**16   # attention for the 8 bit images.
        [input_baseline] = self.normalize([input_baseline])
        mask = mask.permute(2, 0, 1)
        # Log transform
        if self.args.LOG_TRANSFORM:
            input_data = [torch.log2(i+1) for i in input_data]
            
        data = {
            'mask': mask,
            'input': input_data,
            'gt': input_gt,
            'baseline': input_baseline,
            'input_type': input_type,
            'gt_type': gt_type,
            'gt_code': self.args[gt_type],
            'prefix': item_info['prefix'],
            'condition': self._condition_query[item_info['prefix']],
            'gt_names': gt_file_name,    # THIS IS not a list, STRING.
            'baseline_name': baseline_name,
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
        
        # condition
        query = {i: k for k, v in self.condition.items() for i in v}
        count = {k: 0 for k, _ in self.condition.items()}
        for i in self.paired_single_items:
            count[query[i[1]]] += 1
        temp = ''.join([f'{k}:{v}\n' for k, v in count.items()])
        return out_put + temp

    def define_mask(self, gt_img, a_gt_img, gt_code, a_gt_code, mode='', coors=None):
        """
        return: mask, gt_img; gt_img range is (0, 2^16), dtype is float32, tensor
        """
        mask = torch.zeros((self.args.IMG_SIZE[1], self.args.IMG_SIZE[1], self.args.MODALITY_NUM), dtype=torch.float32)
        gt_img = np.array(gt_img).astype('float32')
        a_gt_img = np.array(a_gt_img).astype('float32')
        if coors is not None:
            x, y, w, h = coors
        else:
            x, y = random.randint(0, self.args.IMG_SIZE[1]), random.randint(0, self.args.IMG_SIZE[1])
            w, h = random.randint(0, self.args.IMG_SIZE[1]-x), random.randint(0, self.args.IMG_SIZE[1]-y)
        if mode == 'overwrite':
            mask[..., gt_code] = 1.0
            mask[y:y+h, x:x+w, gt_code] = 0.0
            mask[y:y+h, x:x+w, a_gt_code] = 1.0
            gt_img[y:y+h, x:x+w] = a_gt_img[y:y+h, x:x+w]
        elif mode == 'mean':
            mask[..., gt_code] = 1.0
            mask[y:y+h, x:x+w, gt_code] = 0.5
            mask[y:y+h, x:x+w, a_gt_code] = 0.5
            gt_img[y:y+h, x:x+w] = gt_img[y:y+h, x:x+w]*0.5 + a_gt_img[y:y+h, x:x+w]*0.5
        else:
            raise NotImplementedError(f'{mode} is not implemented ...')
        gt_img = torch.from_numpy(gt_img)[None]
        return mask, gt_img

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