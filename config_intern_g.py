config_dict = {
    'AMP_TYPE': 'float16',
    'AUG': {
        'AUTO_AUGMENT': 'rand-m9-mstd0.5-inc1',
        'COLOR_JITTER': 0.4,
        'CUTMIX': 0.0,
        'CUTMIX_MINMAX': None,
        'MEAN': (0.485, 0.456, 0.406),
        'MIXUP': 0.0,
        'MIXUP_MODE': 'batch',
        'MIXUP_PROB': 1.0,
        'MIXUP_SWITCH_PROB': 0.5,
        'RANDOM_RESIZED_CROP': False,
        'RECOUNT': 1,
        'REMODE': 'pixel',
        'REPROB': 0.0,
        'STD': (0.229, 0.224, 0.225)
    },
    'BASE': [''],
    'DATA': {
        'BATCH_SIZE': 128,
        'CACHE_MODE': 'part',
        'DATASET': 'imagenet',
        'DATA_PATH': '',
        'IMG_ON_MEMORY': True,
        'IMG_SIZE': 512,
        'INTERPOLATION': 'bicubic',
        'NUM_WORKERS': 8,
        'PIN_MEMORY': True,
        'ZIP_MODE': False
    },
    'EVAL_22K_TO_1K': False,
    'EVAL_FREQ': 1,
    'EVAL_MODE': True,
    'LOCAL_RANK': 0,
    'MODEL': {
        'DROP_PATH_RATE': 0.4,
        'DROP_PATH_TYPE': 'linear',
        'DROP_RATE': 0.0,
        'INTERN_IMAGE': {
            'CENTER_FEATURE_SCALE': True,
            'CHANNELS': 512,
            'CORE_OP': 'DCNv3',
            'DEPTHS': [2, 2, 48, 4],
            'DW_KERNEL_SIZE': 5,
            'GROUPS': [16, 32, 64, 128],
            'LAYER_SCALE': None,
            'LEVEL2_POST_NORM': True,
            'LEVEL2_POST_NORM_BLOCK_IDS': [5, 11, 17, 23, 29, 35, 41, 47],
            'MLP_RATIO': 4.0,
            'OFFSET_SCALE': 1.0,
            'POST_NORM': True,
            'REMOVE_CENTER': False,
            'RES_POST_NORM': False,
            'USE_CLIP_PROJECTOR': True
        },
        'LABEL_SMOOTHING': 0.3,
        'NAME': 'internimage_g_22kto1k_512',
        'NUM_CLASSES': 1000,
        'PRETRAINED': '',
        'RESUME': 'models/internimage_g_22kto1k_512.pth',
        'TYPE': 'intern_image'
    },
    'OUTPUT': 'output/internimage_g_22kto1k_512',
    'PRINT_FREQ': 10,
    'SAVE_CKPT_NUM': 1,
    'SAVE_FREQ': 1,
    'SEED': 0,
    'TAG': 'default',
    'TEST': {
        'CROP': True,
        'SEQUENTIAL': False
    },
    'THROUGHPUT_MODE': False,
    'TRAIN': {
        'ACCUMULATION_STEPS': 1,
        'AUTO_RESUME': True,
        'BASE_LR': 5e-06,
        'CLIP_GRAD': 5.0,
        'EMA': {
            'DECAY': 0.9999,
            'ENABLE': True
        },
        'EPOCHS': 20,
        'LR_LAYER_DECAY': True,
        'LR_LAYER_DECAY_RATIO': 0.9,
        'LR_SCHEDULER': {
            'DECAY_EPOCHS': 30,
            'DECAY_RATE': 0.1,
            'NAME': 'cosine'
        },
        'MIN_LR': 0.0,
        'OPTIMIZER': {
            'BETAS': (0.9, 0.999),
            'DCN_LR_MUL': 0.1,
            'EPS': 1e-08,
            'FREEZE_BACKBONE': None,
            'MOMENTUM': 0.9,
            'NAME': 'adamw',
            'USE_ZERO': False
        },
        'RAND_INIT_FT_HEAD': False,
        'START_EPOCH': 0,
        'USE_CHECKPOINT': True,
        'WARMUP_EPOCHS': 2,
        'WARMUP_LR': 0.0,
        'WEIGHT_DECAY': 0.05
    },
    'using core type': 'DCNv3',
    'using activation layer': 'GELU',
    'using main norm layer': 'LN',
    'using dpr': 'linear, 0.4',
    'level2_post_norm': True,
    'level2_post_norm_block_ids': [5, 11, 17, 23, 29, 35, 41, 47],
    'res_post_norm': False,
    'remove_center': False
}