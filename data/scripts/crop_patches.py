# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/20 11:21
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：crop_patches.py
@IDE ：PyCharm
crop patches from the original big images
"""
from PIL import Image
import numpy as np
import os


def crop_images(img, size, redundant):
    h, w = img.shape
    results = {}
    for i in range(0, h, redundant):
        for j in range(0, w, redundant):
            if i+size < h and j+size < w:
                patch = img[i:i+size, j:j+size].copy()
                results[f'{i}_{j}_{size}_{size}'] = patch
    return results

def process(args):
    n_index, name = args
    print('Labs:[{}/{}], Conditions:[{}/{}], Images: [{}/{}], img_name:{}'.format(
        l_index+1, len(labs), c_index+1, len(conditions), n_index+1, len(img_names),
        name
    ))
    img_path = os.path.join(img_dir, name)
    img = Image.open(img_path)
    img = np.array(img)
    results = crop_images(img, size, redundant)
    for location, img in results.items():
        save_path = os.path.join(save_dir, '{},{}.png'.format(name, location))
        img = Image.fromarray(img)
        img.save(save_path)


if __name__ == '__main__':
    import argparse
    from multiprocessing.dummy import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--root', type=str, default='../ISL/')
    args = parser.parse_args()
    flag = args.data_type

    save_flag = 'test'
    root = os.path.join(args.root, '{}_single_channel_images'.format(flag))
    save_root = '../ISL/patches_{}'.format(save_flag)


    size = 256
    redundant = 128
    labs = os.listdir(root)
    for l_index, lab in enumerate(labs):
        lab_dir = os.path.join(root, lab)
        conditions = os.listdir(lab_dir)
        for c_index, condition in enumerate(conditions):
            img_dir = os.path.join(lab_dir, condition)
            img_names = os.listdir(img_dir)
            save_dir = os.path.join(save_root, lab, condition)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            args = [(i, n) for i, n in enumerate(img_names)]
            p = Pool(16)
            p.map(process, args)

            # for n_index, name in enumerate(img_names):
            #     print('Labs:[{}/{}], Conditions:[{}/{}], Images: [{}/{}], img_name:{}'.format(
            #         l_index+1, len(labs), c_index+1, len(conditions), n_index+1, len(img_names),
            #         name
            #     ))
            #     img_path = os.path.join(img_dir, name)
            #     img = Image.open(img_path)
            #     img = np.array(img)
            #     results = crop_images(img, size, redundant)
            #     for location, img in results.items():
            #         save_path = os.path.join(save_dir, '{},{}.png'.format(name, location))
            #         img = Image.fromarray(img)
            #         img.save(save_path)



