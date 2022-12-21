import json
import os


def is_file_exists(root, info_item):
    coors = [str(i) for i in info_item['coors']]
    prefix = info_item['prefix']
    for k, v in info_item['labels'].items():
        p = os.path.join(root, prefix, v+','+'_'.join(coors)+'.png')
        if not os.path.exists(p):
            print(p)
            return False
    for k, v in info_item['inputs'].items():
        p = os.path.join(root, prefix, v + ',' + '_'.join(coors) + '.png')
        if not os.path.exists(p):
            print(p)
            return False
    for k, v in info_item['baseline'].items():
        p = os.path.join(root, prefix, v+','+'_'.join(coors) + '.png')
        if not os.path.exists(p):
            print(p)
            return False
    return True


if __name__ == '__main__':
    flag = 'test'
    save_root = '../ISL/patches_{}'.format(flag)
    json_path = '../ISL/patches_{}.json'.format(flag)
    with open(json_path) as f:
        all_items = json.load(f)

    for item in all_items:
        status = is_file_exists(save_root, item)
        if not status:
            raise RuntimeError('file do not exists...')
