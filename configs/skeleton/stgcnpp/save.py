# ===================================================================
# STGCN++ 3-Class Golf Action Recognition Config - GENERALIZATION V1.2
# ===================================================================

# ------------------------ Base Configuration ------------------------
_base_ = '../../_base_/default_runtime.py'
default_scope = 'mmaction'

# ------------------------ Tunable hyperparameters ------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 0.0005 # 학습률 하향 조정
WEIGHT_DECAY = 0.001 # L2 정규화 강화
MAX_EPOCHS = 100
PATIENCE = 10
WARMUP_EPOCHS = 5
FEATS='b'

# ------------------------ Path Configuration ------------------------
_load_checkpoint_path = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth"
dataset_type = 'PoseDataset'
ann_file = r"E:\golfDataset\dataset\crop_pkl\combined_3class.pkl"
test_ann_file = r"D:\golfDataset\dataset\crop_pkl\combined_3class_test.pkl"
EPOCH = MAX_EPOCHS
clip_len = 100

# ------------------------ Data Pipeline (Train) ------------------------
train_pipeline = [
    dict(type='PreNormalize2D'),
    
    # 🚨 RandomShift 제거 (KeyError 발생으로 인한 조치)
    dict(
        type='RandomAffine',
        scale_range=(0.8, 1.2), # 스케일 변화
        shift_range=(-0.1, 0.1), # 위치 변화
        rotate_range=(-15, 15), # 회전 변화
        shear_range=(0, 0),
        p=0.5
    ),

    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]
    ),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

# ------------------------ Data Pipeline (Val/Test) ------------------------
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    dict(
        type='UniformSampleFrames',
        clip_len=clip_len,
        num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(
        type='Collect',
        keys=('keypoint', 'label'),
        meta_keys=('frame_interval', 'label_index')
    )
]

# TTA를 적용한 최종 테스트 파이프라인
test_pipeline = [
    # 10 clips 샘플링
    dict(type='LoadPose'),
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    dict(
        type='UniformSampleFrames',
        clip_len=clip_len,
        num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    # TTA: 원본 + Flip (좌우 반전) 두 가지 view를 테스트
    dict(type='Collect', keys=('keypoint', 'label'), meta_keys=('frame_interval', 'label_index')),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False,
         target_shape=(56, 56), flip=True) # flip=True로 TTA 활성화
]


# ------------------------ Data Loader & Loop ------------------------
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train')))

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCH, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------------ Learning Rate Scheduler ------------------------
# CosineAnnealingLR로 변경하여 수렴 품질 개선
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=WARMUP_EPOCHS),
    dict(
        type='CosineAnnealingLR',
        T_max=EPOCH - WARMUP_EPOCHS,
        by_epoch=True,
        begin=WARMUP_EPOCHS,
        end=EPOCH)
]

# ------------------------ Optimizer ------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False
    ),
    clip_grad=dict(max_norm=2, norm_type=2))

# ------------------------ Model & Evaluator ------------------------
val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial'),
        init_cfg=dict(type='Pretrained', checkpoint=_load_checkpoint_path)
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=3,
        in_channels=256,
        dropout=0.5, 
        loss_cls=dict(
            type='CBFocalLoss',
            loss_weight=1.0,
            # 클래스 0, 1의 가중치를 더 극단적으로 높여 분리 강제
            samples_per_cls=[500, 100, 2431],
            beta=0.9999,
            gamma=2.0
        )
    )
)