from .discriminator import PatchD, ConvD
from .caswinvs import  MAEUSCASwinVS


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'maeuscaswinvs':
        # stupid code for multi dataset support
        use_decode_mask = False if config.DATA.DATASET == 'hedataset' else True
        model = MAEUSCASwinVS(
                        img_size=config.MODEL.INPUT_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        out_chans=config.MODEL.SWIN.OUT_CHANS,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        use_channel_attention=config.MODEL.SWIN.CHANNEL_ATTENTION,
                        pos_embed_mod=config.MODEL.SWIN.POS_EMBED_MODE,
                        is_pretrain=config.MODEL.SWIN.IS_PRETRAIN,
                        decoder_depth=config.MODEL.SWIN.DECODER_DEPTH,
                        decoder_embed_dim=config.MODEL.SWIN.DECODER_EMBED_DIM,
                        us_window=config.MODEL.SWIN.PRETRAIN_US_WINDOW,
                        is_resize_feature_maps=config.MODEL.SWIN.IS_RESIZE_FEATURE_MAPS,
                        mask_ratio=config.MODEL.SWIN.MASK_RATIO,
                        use_decode_mask=use_decode_mask,
                        head_type=config.MODEL.SWIN.HEAD_TYPE,
        )    

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_discriminator(model_type='ConvD', img_size=128, data_type='silicio'):
    if model_type == 'PatchD':
        model = PatchD(8)
    elif model_type == 'ConvD':
        if data_type == 'silicio':
            n = 9
        else:
            n = 3
        model = ConvD(n)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
