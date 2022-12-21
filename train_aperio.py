# --------------------------------------------------------
# Different with main_gan.py
# This version use multi-head in the recstruction module

# Written by Ze Liu, modified by jiabo ma
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
from torchvision import utils as vutils
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from timm.utils import AverageMeter
from metrics import easy_psnr
from loss import PhaseLoss, CrossEntropy, SoftTargetCrossEntropy, CharBonnierLoss

from config import get_config
from models import build_model, build_discriminator
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer, build_optimizer_d
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
    

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_loader(config)
    use_amp = config.AMP_OPT_LEVEL
    if use_amp != 'O0':
        print('AMP is enabled ....')
        grad_scaler = amp.grad_scaler.GradScaler()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model_d = build_discriminator('ConvD', data_type=config.DATA.DATASET)
    model.cuda()
    model_d.cuda()
    logger.info(str(model))
    logger.info(str(model_d))
    if dist.get_rank() == 0:
        sm = SummaryWriter(config.OUTPUT)
    else:
        sm = None
    optimizer = build_optimizer(config, model)
    optimizer_d = build_optimizer_d(config, model_d)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_d = torch.nn.parallel.DistributedDataParallel(model_d, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    model_d_without_ddp = model_d.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    lr_scheduler_d = build_scheduler(config, optimizer_d, len(data_loader_train))

    rec_loss = {'ce': CrossEntropy(), 'l1': nn.L1Loss(), 'smoothl1': nn.SmoothL1Loss(), 
                'stce': SoftTargetCrossEntropy().cuda(), 'ch': CharBonnierLoss(),
                'mse': nn.MSELoss(),
                }
    criterion = {'rec': rec_loss[config.TRAIN.REC_LOSS_NAME], 'bce': nn.BCELoss(), 
                'phase': PhaseLoss('cosin'), 'mse': nn.MSELoss()}
    for k, v in criterion.items():
        print(f'{k}:{v}')

    max_psnr = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_psnr = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger,
                    model_d=model_d_without_ddp, optimizer_d=optimizer_d, 
                    lr_scheduler_d=lr_scheduler_d, scaler=grad_scaler)
        psnr, loss = validate(config, data_loader_val, model)
        logger.info(f"PSNR of the network on the {len(dataset_val)} test images: {psnr:.3f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        # psnr, loss = validate(config, data_loader_val, model)
        # logger.info(f"PSNR of the network on the {len(dataset_val)} test images: {psnr:.3f}%")


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, model_d, criterion, data_loader_train, optimizer, 
                        optimizer_d, epoch, lr_scheduler, sm, grad_scaler=grad_scaler,
                        use_amp=use_amp)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            model_d_temp = model_d_without_ddp if epoch > config.TRAIN.GAN.START_EPOCH else None
            optimizer_d_temp = optimizer_d if epoch > config.TRAIN.GAN.START_EPOCH else None
            lr_scheduler_d_temp = lr_scheduler_d if epoch > config.TRAIN.GAN.START_EPOCH else None
            save_checkpoint(config, epoch, model_without_ddp, max_psnr, optimizer, lr_scheduler, logger,
                            model_d_temp, optimizer_d_temp, lr_scheduler_d_temp, grad_scaler)

        psnr, loss = validate(config, data_loader_val, model, epoch, sm)
        logger.info(f"PSNR of the network on the {len(dataset_val)} test images: {psnr:.3f}%")
        max_psnr = max(max_psnr, psnr)
        logger.info(f'Max PSNR: {max_psnr:.3f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, model_d, criterion, data_loader, optimizer, optimizer_d, epoch, lr_scheduler, 
                    summary_writer, grad_scaler=None, use_amp=True):
    model.train()
    model_d.train()
    optimizer.zero_grad()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    total_loss_g_meter = AverageMeter()
    l1_loss_meter = AverageMeter()
    gan_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    phase_meter = AverageMeter()

    norm_d_meter = AverageMeter()
    total_loss_d_meter = AverageMeter()
    d_real_meter = AverageMeter()
    d_fake_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        small, middle, big = data['he']
        small, middle, big = small.cuda(non_blocking=True), middle.cuda(non_blocking=True), big.cuda(non_blocking=True)
        targets = data['ihc'][1].cuda(non_blocking=True)
        # convert to (-1, 1)
        small, middle, big = (small - .5)*2, (middle - .5)*2, (big - .5)*2
        targets = (targets - .5)*2

        total = 0
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, _, _ = model(small, middle, big)
                l1_loss = criterion['rec'](outputs, targets)
                total += l1_loss
                # phase loss
                if config.TRAIN.PHASE_LOSS_RATIO > 0:
                    phase_loss = criterion['phase'](outputs, targets)*config.TRAIN.PHASE_LOSS_RATIO
                    total += phase_loss
                # GAN
                if epoch > config.TRAIN.GAN.START_EPOCH:
                    if config.TRAIN.GAN.TYPE == 'vanilla':
                        predict_label = model_d(outputs)
                        loss_fn = criterion['bce']
                        gan_loss = loss_fn(predict_label, torch.ones_like(predict_label))*config.TRAIN.GAN.G_GAN_RATIO
                    elif config.TRAIN.GAN.TYPE == 'lsgan':
                        predict_label = model_d(outputs)
                        loss_fn = criterion['mse']
                        gan_loss = loss_fn(predict_label, torch.ones_like(predict_label))*config.TRAIN.GAN.G_GAN_RATIO
                    else:
                        raise NotImplementedError('{} is not implemented ...'.format(config.TRAIN.GAN.TYPE))
                    total += gan_loss
            optimizer.zero_grad()
            grad_scaler.scale(total).backward()
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        lr_scheduler.step_update(epoch * num_steps + idx)
        # discriminator part
        if epoch > config.TRAIN.GAN.START_EPOCH and idx % 4 ==0:
            if config.TRAIN.GAN.TYPE == 'vanilla':
                loss_fn = criterion['bce']
            elif config.TRAIN.GAN.TYPE == 'lsgan':
                loss_fn = criterion['mse']

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                real_label = model_d(targets)
                real_loss = loss_fn(real_label, torch.ones_like(real_label))
                fake_label = model_d(outputs.detach())
                fake_loss = loss_fn(fake_label, torch.zeros_like(fake_label))
                total_d  = (real_loss + fake_loss)/2

                optimizer_d.zero_grad()
                grad_scaler.scale(total_d).backward()
                grad_scaler.unscale_(optimizer_d)
                grad_norm_d = torch.nn.utils.clip_grad_norm_(model_d.parameters(), config.TRAIN.CLIP_GRAD)
                grad_scaler.step(optimizer_d)
                grad_scaler.update()                

        torch.cuda.synchronize()
        # Generator part
        total_loss_g_meter.update(total.item(), targets.size(0))
        l1_loss_meter.update(l1_loss.item(), targets.size(0))
        if epoch > config.TRAIN.GAN.START_EPOCH:
            gan_loss_meter.update(gan_loss.item(), targets.size(0))
        if config.TRAIN.PHASE_LOSS_RATIO > 0:
            phase_meter.update(phase_loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        # Dis part
        if epoch > config.TRAIN.GAN.START_EPOCH:
            total_loss_d_meter.update(total_d.item(), targets.size(0))
            d_real_meter.update(real_loss.item(), targets.size(0))
            d_fake_meter.update(fake_loss.item(), targets.size(0))
            norm_d_meter.update(grad_norm_d)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'l1 loss {l1_loss_meter.val:.4f} ({l1_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            # save images
            if dist.get_rank() == 0 and idx % (config.PRINT_FREQ*10) == 0:
                img = torch.cat([outputs, targets], dim=-1)
                path = os.path.join(config.OUTPUT, 'train_img', f'{epoch}_{idx}.png')
                vutils.save_image(img, path, normalize=True, range=(-1, 1))

                summary_writer.add_scalar('train/G loss', total_loss_g_meter.avg, epoch*len(data_loader)+idx)
                summary_writer.add_scalar('train/l1 loss', l1_loss_meter.avg, epoch*len(data_loader)+idx)
                if config.TRAIN.PHASE_LOSS_RATIO > 0:
                    summary_writer.add_scalar('train/phase loss', phase_meter.avg, epoch*len(data_loader)+idx)

                if epoch > config.TRAIN.GAN.START_EPOCH:
                    summary_writer.add_scalar('train/D loss', total_loss_d_meter.avg, epoch*len(data_loader)+idx)
                    summary_writer.add_scalar('train/real loss', d_real_meter.avg, epoch*len(data_loader)+idx)
                    summary_writer.add_scalar('train/fake loss', d_fake_meter.avg, epoch*len(data_loader)+idx)
                    summary_writer.add_scalar('train/GAN loss', gan_loss_meter.avg, epoch*len(data_loader)+idx)

                summary_writer.add_scalar('train/lr', lr, epoch*len(data_loader)+idx)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch=None, summary_writer=None, use_amp=True):
    criterion = torch.nn.L1Loss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    end = time.time()
    for idx, data in enumerate(data_loader):
        small, middle, big = data['he']
        small, middle, big = small.cuda(non_blocking=True), middle.cuda(non_blocking=True), big.cuda(non_blocking=True)
        target = data['ihc'][1].cuda(non_blocking=True)
        # convert to (-1, 1)
        small, middle, big = (small - .5)*2, (middle - .5)*2, (big - .5)*2
        target = (target - .5)*2
        # compute output
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output, _, _ = model(small, middle, big)
                output = (output + 1)/2
                target = (target + 1)/2

        loss = criterion(output, target)
        psnr = easy_psnr(output, target)

        psnr = reduce_tensor(psnr)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        psnr_meter.update(psnr.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'PSNR {psnr_meter.val:.3f} ({psnr_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * PSNR {psnr_meter.avg:.3f}')
    if summary_writer is not None:
        summary_writer.add_scalar('test/psnr', psnr_meter.avg, epoch)
    return psnr_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size()
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size()
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size()
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT, 'train_img'), exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
