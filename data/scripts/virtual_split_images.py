# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/20 11:21
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：crop_patches.py
@IDE ：PyCharm
split the big image to patches, output a json index file.
"""
import json
from PIL import Image
import os
import argparse


def split_images(path, size, redundant):
    img = Image.open(path)
    w, h = img.size
    results = []
    for i in range(0, h, redundant):
        for j in range(0, w, redundant):
            if i+size < h and j+size < w:
                results.append((i, j, size, size))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--root', type=str, default='../ISL/')

    args = parser.parse_args()
    flag = args.data_type
    root = os.path.join(args.root, '{}_single_channel_images'.format(flag))

    save_path = '../ISL/patches_{}.json'.format(flag)
    results = []
    size = 256
    redundant = 128

    labs = os.listdir(root)
    for l_index, lab in enumerate(labs):
        lab_dir = os.path.join(root, lab)
        conditions = os.listdir(lab_dir)
        for c_index, condition in enumerate(conditions):
            img_dir = os.path.join(lab_dir, condition)
            img_names = os.listdir(img_dir)
            prefix = [','.join(i.split(',')[:7])for i in img_names]
            prefix_set = list(set(prefix))
            for n_index, name in enumerate(prefix_set):
                print('Labs:[{}/{}], Conditions:[{}/{}], Images: [{}/{}], img_name:{}'.format(
                    l_index+1, len(labs), c_index+1, len(conditions), n_index+1, len(img_names),
                    name
                ))
                matched_img_names = [n for n in img_names if name in n]
                img_path = os.path.join(img_dir, matched_img_names[0])
                coors = split_images(img_path, size, redundant)
                for c in coors:
                    gt_names = [i for i in matched_img_names if ('computation' in i and 'ORIGINAL' in i)]
                    gt_types = [i.split(',')[10].split('-')[-1] for i in gt_names]
                    baseline_names = [i for i in matched_img_names if ('computation' in i and 'PREDICTED' in i)]
                    baseline_types = [i.split(',')[10].split('-')[-1] for i in baseline_names]
                    input_names = [i for i in matched_img_names if 'z_depth' in i]
                    item = {
                        'labels': {k: v for k, v in zip(gt_types, gt_names)},
                        'inputs': {k.split(',')[7].split('-')[-1]: k for k in input_names},
                        'input_type': input_names[0].split(',')[9].split('-')[-1],
                        'prefix': f'{lab}/{condition}',
                        'coors': c,
                        'baseline': {k: v for k, v in zip(baseline_types, baseline_names)},
                    }
                    results.append(item)

    with open(save_path, 'w') as f:
        json.dump(results, f)




