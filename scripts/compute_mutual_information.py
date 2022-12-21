from sklearn import metrics
import json
import os
import cv2


root = '/home/jmabq/projects/Swin-Transformer/output/caswinvs_base_patch1_window8_128_6_6/BCIdata_maeus_train_resize_us2_mask50/test/ckpt_epoch_240/test_img_split'
save_txt = '/home/jmabq/projects/Swin-Transformer/output/caswinvs_base_patch1_window8_128_6_6/BCIdata_maeus_train_resize_us2_mask50/test/ckpt_epoch_240/mi.txt'
size = 128

def query_condition(file_name):
    items = file_name.split(',')
    prefix = items[0].split('-')[-1] + '/' + items[1].split('-')[-1]
    condition = {
        'A': ['Rubin/scott_1_0'],
        'B': [f'Finkbeiner/kevan_0_{i}' for i in [7, 8, 9, 10]],
        'C': ['Finkbeiner/alicia_2_0'],
        'D': ['Finkbeiner/yusha_0_1'],
        'E': ['GLS/2015_06_26']
    }
    condition_query = {i:k for k, v in condition.items() for i in v}
    d = condition_query[prefix]
    return d



if __name__ == '__main__':

    mi_list = {}
    all_files = os.listdir(root)
    for index, f in enumerate(all_files):
        print('progress:{}/{}'.format(index, len(all_files)))
        # c = query_condition(f)
        c = 'all'
        if c == 'E':
            continue
        mi_list.setdefault(c, [])
        p = os.path.join(root, f)
        img = cv2.imread(p, flags=0)
        # predict = img[:, :size]
        # gt = img[:, size:size*2]
        # adjust based on the position of predict and gt
        predict = img[:, size:size*2]
        gt = img[:, -size:]
        # convrt to int
        gt = gt.reshape(-1).astype('int')
        predict = predict.reshape(-1).astype('int')
        mi = metrics.normalized_mutual_info_score(gt, predict)
        mi_list[c].append(mi)
    with open(save_txt, 'w') as f:
        total = []
        for c, values in mi_list.items():
            f.write('{} MI:  {}\n'.format(c, sum(values)/len(values)))
            total.extend(values)
        f.write('AVG MI:{}\n'.format(sum(total)/len(total)))
        f.write('numbers:{}\n'.format(len(total)))
