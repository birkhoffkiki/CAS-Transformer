
import os
import cv2
import torch

from metrics import easy_psnr, easy_psnr_ssim
from loss import mask2image

from config import get_config_wo_cmdline
from models import build_model
from data import build_test_loader
from logger import create_logger


@torch.no_grad()
def validate(config, data_loader, model, save_baseline=True, output_num=1):
    model.eval()

    eval_psnr, eval_ssim = [], []
    logger.info('-----Details START----')
    for idx, data in enumerate(data_loader):
        print('progress: {}/{}'.format(idx, len(data_loader)))
        
        small, middle, big = data['he']
        small, middle, big = small.cuda(non_blocking=True), middle.cuda(non_blocking=True), big.cuda(non_blocking=True)
        target = data['ihc'][1]
        name = data['name']
        # convert to (-1, 1)
        small, middle, big = (small - .5)*2, (middle - .5)*2, (big - .5)*2
        target = (target - .5)*2

        predict, _, _ = model(small, middle, big)
        predict = (predict + 1)/2
        target = (target + 1)/2
        middle = (middle + 1)/2

        predict = predict.cpu()
        target = target.cpu()
        middle = middle.cpu()
        for i in range(len(target)):
            p = predict[i][None]
            g = target[i][None]
            _psnr, _ssim = easy_psnr_ssim(p, g)
            eval_psnr.extend(_psnr)
            eval_ssim.extend(_ssim)
            # save images;
            img = torch.cat([middle[i][None], p, g], dim=-1)[0]
            img = img.permute(1, 2, 0).clip_(0, 1.0).numpy()*255
            path = os.path.join(output, 'test_img', name[i])
            cv2.imwrite(path, img.astype('uint8'))

    logger.info('-----Details END----')
    logger.info('\t >> PSNR:{}\n'.format(sum(eval_psnr)/len(eval_psnr)))
    logger.info('\t >> SSIM:{}\n'.format(sum(eval_ssim)/len(eval_ssim)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # config file path
    cfg_path = './configs/AperioData/train.yaml'
    # which checkpoint file, e.g. ckpt_epoch_20.pth
    ckpt = './weights/aperio.pth'
    model_output_num = 3 # the output number of model


    config = get_config_wo_cmdline(cfg_path)

    output = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG, 'test', ckpt.split('.')[0])
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'test_img'), exist_ok=True)
    logger = create_logger(output_dir=output, dist_rank=0, name=f"{config.MODEL.NAME}")
    # print config
    logger.info(config.dump())
    dataset_val, data_loader_val = build_test_loader(config)
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
    info = model.load_state_dict(torch.load(resume_file))
    print(info)
    validate(config, data_loader_val, model, output_num=model_output_num)


