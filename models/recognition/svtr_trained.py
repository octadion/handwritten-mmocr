default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dictionary = dict(
    dict_file=
    'models/recognition/lower_english_digits.txt',
    type='Dictionary',
    with_padding=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2015_1811_textrecog_test = dict(
    ann_file='textrecog_test_1811.json',
    data_root='/content/dataset-rec',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_data_root = '/content/dataset-rec'
icdar2015_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/content/dataset-rec',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/content/dataset-rec',
    pipeline=None,
    type='OCRDataset')
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    data_preprocessor=dict(
        mean=[
            127.5,
        ], std=[
            127.5,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file=
            'models/recognition/lower_english_digits.txt',
            type='Dictionary',
            with_padding=True,
            with_unknown=True),
        in_channels=256,
        module_loss=dict(
            letter_case='lower', type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        type='SVTRDecoder'),
    encoder=dict(
        depth=[
            3,
            6,
            9,
        ],
        embed_dims=[
            128,
            256,
            384,
        ],
        img_size=[
            48,
            160,
        ],
        in_channels=3,
        max_seq_len=40,
        merging_types='Conv',
        mixer_types=[
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
        ],
        num_heads=[
            4,
            8,
            12,
        ],
        out_channels=256,
        prenorm=False,
        type='SVTREncoder',
        window_size=[
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
        ]),
    preprocessor=dict(
        in_channels=3,
        margins=[
            0.05,
            0.05,
        ],
        num_control_points=20,
        output_image_size=(
            48,
            160,
        ),
        resized_image_size=(
            32,
            64,
        ),
        type='STN'),
    type='SVTR')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.99,
        ),
        eps=8e-08,
        lr=0.0005,
        type='AdamW',
        weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        convert_to_iter_based=True,
        end=2,
        end_factor=1.0,
        start_factor=0.5,
        type='LinearLR',
        verbose=False),
    dict(
        T_max=19,
        begin=2,
        convert_to_iter_based=True,
        end=20,
        type='CosineAnnealingLR',
        verbose=False),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='/content/dataset-rec',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='Evaluator')
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='/content/dataset-rec',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='/content/dataset-rec',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='TextRecogGeneralAug'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='CropHeight'),
                ],
                type='RandomApply'),
            dict(
                condition='min(results["img_shape"])>10',
                true_transforms=dict(
                    prob=0.4,
                    transforms=[
                        dict(
                            kernel_size=5,
                            op='GaussianBlur',
                            sigma=1,
                            type='TorchVisionWrapper'),
                    ],
                    type='RandomApply'),
                type='ConditionApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        brightness=0.5,
                        contrast=0.5,
                        hue=0.1,
                        op='ColorJitter',
                        saturation=0.5,
                        type='TorchVisionWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ImageContentJitter'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        args=[
                            dict(
                                cls='AdditiveGaussianNoise',
                                scale=0.31622776601683794),
                        ],
                        type='ImgAugWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ReversePixels'),
                ],
                type='RandomApply'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='/content/dataset-rec',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug'),
        ],
        type='RandomApply'),
    dict(prob=0.4, transforms=[
        dict(type='CropHeight'),
    ], type='RandomApply'),
    dict(
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            prob=0.4,
            transforms=[
                dict(
                    kernel_size=5,
                    op='GaussianBlur',
                    sigma=1,
                    type='TorchVisionWrapper'),
            ],
            type='RandomApply'),
        type='ConditionApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                brightness=0.5,
                contrast=0.5,
                hue=0.1,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                args=[
                    dict(
                        cls='AdditiveGaussianNoise',
                        scale=0.31622776601683794),
                ],
                type='ImgAugWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ReversePixels'),
        ],
        type='RandomApply'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    256,
                    64,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='/content/dataset-rec',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='Evaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/content/work_dir'
