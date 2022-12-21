import os
import torch
import torch.distributed as dist

def reset_epochs(scheduler, decay_steps, decay):
    scheduler.decay_t = decay_steps
    scheduler.decay_rate=decay
    return scheduler


def load_checkpoint(config, model, optimizer, lr_scheduler, logger,
                    model_d=None, optimizer_d=None, lr_scheduler_d=None, scaler=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    # dis
    try:
        if 'model_d' in checkpoint and checkpoint['model_d'] is not None:
            model_d.load_state_dict(checkpoint['model_d'])
        if 'optimizer_d' in checkpoint and checkpoint['optimizer_d'] is not None:
            # pass 
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        if 'lr_scheduler_d' in checkpoint and checkpoint['lr_scheduler_d'] is not None:
            lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
    except:
        print('Failed to load ...')

    max_psnr = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0":
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_psnr' in checkpoint:
            max_psnr = checkpoint['max_psnr']

    del checkpoint
    torch.cuda.empty_cache()
    return max_psnr


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']
    # discard last head
    # state_dict.pop('conv_last.weight')
    # state_dict.pop('conv_last.bias')

    try:
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
    except:
        # remove keys including 'attn_mask', for maeuscaswin model if use pretrain
        _keys = []
        for k in state_dict.keys():
            if 'attn_mask' in k:
                _keys.append(k)
        for k in _keys:
            print(f'{k} is discarded !!!')
            state_dict.pop(k)
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        print('Load all keys from pretrained files....')
        
    logger.warning(msg)
    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, 
                    model_d=None, optimizer_d=None, lr_scheduler_d=None, scaler=None):
    model_d = model_d.state_dict() if model_d is not None else None
    optimizer_d = optimizer_d.state_dict() if optimizer_d is not None else None
    lr_scheduler_d = lr_scheduler_d.state_dict() if lr_scheduler_d is not None else None
    lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler_state,
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config,
                  'model_d': model_d,
                  'optimizer_d': optimizer_d,
                  'lr_scheduler_d': lr_scheduler_d
                  }
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = scaler.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
