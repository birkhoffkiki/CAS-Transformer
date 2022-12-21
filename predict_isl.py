import os
import cv2
import torch
from metrics import easy_psnr_ssim

from config import get_config_wo_cmdline
from models import build_model
from data import build_test_loader
from logger import create_logger


@torch.no_grad()
def validate(config, data_loader, model, save_baseline=False, output_num=1, save_img=True):
    model.eval()

    eval_psnr, eval_ssim = {}, {}
    condition_psnr, condition_ssim = {}, {}
    total_psnr, total_ssim = [], []
    counter = 0
    logger.info('-----Details START----')
    for idx, data in enumerate(data_loader):
        print('{}/{}'.format(idx, len(data_loader)))
        print(idx, '/', len(data_loader))
        small, middle, big = data['input']
        small, middle, big = small.cuda(non_blocking=True), middle.cuda(non_blocking=True), big.cuda(non_blocking=True)
        mask = data['mask'].cuda(non_blocking=True)
        target = data['gt'].cuda(non_blocking=True)

        baseline = data['baseline']
        condition = data['condition']
        gt_type = data['gt_type']
        input_type = data['input_type']
        gt_file_name = data['gt_names']

        # compute output
        if output_num == 1:
            predict = model(small, middle, big, mask)
        elif output_num == 3:
            predict, _, _ = model(small, middle, big)
            _index = mask.argmax(dim=1)[:, None]
            predict = torch.gather(predict, 1, _index)
        else:
            raise ValueError(f'Not support `output_num = {output_num}`')


        predict = predict.clamp_(0, 1.0)
        predict = predict.cpu()
        target = target.cpu()
        psnr, ssim = easy_psnr_ssim(predict, target)

        for i in range(len(predict)):
            if condition[i] == 'E':
                print('Condition E is detected, skip...')
                continue
            _psnr, _ssim = psnr[i], ssim[i]
            eval_psnr.setdefault(gt_type[i], [])
            eval_ssim.setdefault(gt_type[i], [])
            condition_psnr.setdefault(condition[i], [])
            condition_ssim.setdefault(condition[i], [])

            eval_psnr[gt_type[i]].extend([_psnr])
            eval_ssim[gt_type[i]].extend([_ssim])
            condition_psnr[condition[i]].extend([_psnr])
            condition_ssim[condition[i]].extend([_ssim])
            total_psnr.extend([_psnr])
            total_ssim.extend([_ssim])
            if save_img:
                if save_baseline:
                    img = torch.cat([predict[i], target[i], baseline[i]], dim=-1)
                else:
                    img = torch.cat([predict[i], target[i]], dim=-1)
                img = img.permute(1, 2, 0).clip(0, 1.0).numpy()*255
                # original one, save without image name
                # cv2.imwrite(os.path.join(output, 'test_img', f'{counter}.png'), img.astype('uint8'))
                cv2.imwrite(os.path.join(output, 'test_img', gt_file_name[i]), img.astype('uint8'))
                logger.info('{} : {} > {} > {} > {:.3f} > {:.3f}'.format(
                counter, condition[i], input_type[i], gt_type[i], _psnr, _ssim
                ))
                counter += 1

    logger.info('-----Details END----')
    logger.info('statistical metrics:\n')
    for k in eval_psnr.keys():
        _psnr = eval_psnr[k]
        _ssim = eval_ssim[k]
        avg_ssim = sum(_ssim)/len(_ssim)
        avg_psnr = sum(_psnr)/len(_psnr)
        logger.info(f'{k}({len(_psnr)}):\n\tssim:{avg_ssim}, \n\tpsnr:{avg_psnr}\n')
    logger.info('>>Condition situations:\n')
    for k in condition_psnr.keys():
        _psnr = condition_psnr[k]
        _ssim = condition_ssim[k]
        avg_ssim = sum(_ssim)/len(_ssim)
        avg_psnr = sum(_psnr)/len(_psnr)
        logger.info(f'{k}({len(_psnr)}):\n\tssim:{avg_ssim}, \n\tpsnr:{avg_psnr}\n')
    logger.info('average of all items:\n')
    logger.info('\t >> PSNR:{}\n'.format(sum(total_psnr)/len(total_psnr)))
    logger.info('\t >> SSIM:{}\n'.format(sum(total_ssim)/len(total_ssim)))


if __name__ == '__main__':
    print('PLEASE ATTENTION THR ROOT PATH OF DATA...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # config file path
    cfg_path = './configs/ISL/test.yaml'
    # which checkpoint file, e.g. ckpt_epoch_20.pth
    ckpt = './weights/isl.pth'

    model_output_num = 3 # the output number of model; 3 for ISL dataset
    test_64 = False # redundant 64, used for splice 
    mode = 'test_64' if test_64 else 'test'
    save_img = False
    

    config = get_config_wo_cmdline(cfg_path)
    output = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG, mode)
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'test_img'), exist_ok=True)
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")
    # print config
    logger.info(config.dump())
    dataset_val, data_loader_val = build_test_loader(config, test_64=test_64)
    model = build_model(config)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # load checkpoint
    if os.path.exists(ckpt):
        resume_file = ckpt
    else:
        resume_file = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG, ckpt)

    logger.info('ckpt file: {}'.format(resume_file))
    info = model.load_state_dict(torch.load(resume_file)['model'])
    print(info)
    validate(config, data_loader_val, model, output_num=model_output_num, save_img=save_img)


