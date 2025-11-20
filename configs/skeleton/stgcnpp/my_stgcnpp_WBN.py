# ===================================================================
# STGCN++ 5-Class Golf Action Recognition Config - STABLE_V3.0 (Generalization & Class Imbalance Focus)
# ===================================================================

# ------------------------ Base Configuration ------------------------
_base_ = '../../_base_/default_runtime.py'
default_scope = 'mmaction'

# ------------------------ Tunable hyperparameters (일반화 및 불균형 강화 설정) ------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 0.0005 
WEIGHT_DECAY = 0.0005 
MAX_EPOCHS = 100
PATIENCE = 10
WARMUP_EPOCHS = 5 # 👈 Warmup 연장 (안정적인 학습 시작 유도)
FEATS='j'

# ------------------------ Path Configuration ------------------------
_load_checkpoint_path = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
dataset_type = 'PoseDataset'
ann_file = r"E:\golfDataset\dataset\crop_pkl\combined_WBNclass.pkl" # 5-class 데이터셋으로 가정
test_ann_file = r"D:\golfDataset\dataset\crop_pkl\combined_WBNclass_test.pkl" # 5-class 데이터셋으로 가정
EPOCH = MAX_EPOCHS
clip_len = 100

# ------------------------ Data Pipeline (Train) ------------------------
train_pipeline = [
    dict(type='PreNormalize2D'),
    
    # RandomAffine: 일반화 개선을 위해 범위 복구 (Test Set 분포 포괄)
    dict(
        type='RandomAffine',
        scale_range=(0.8, 1.2), # 👈 범위 확대
        shift_range=(-0.1, 0.1),
        rotate_range=(-15, 15), # 👈 범위 확대
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
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    dict(
        type='UniformSampleFrames',
        clip_len=clip_len,
        num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs') 
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
        num_classes=3, # worst, bad, normal
        in_channels=256,
        dropout=0.5, # 👈 드롭아웃 감소로 과도한 정규화 방지
        loss_cls=dict(
            type='CBFocalLoss',
            loss_weight=1.0,
            # Worst(0)와 Bad(1)에 대한 집중도를 높이기 위해 샘플 수와 감마값 조정
            # 0: 110, 1: 1116, 2: 1041, 3: 788, 4: 1401 (이전 데이터셋의 샘플 수 유지)
            samples_per_cls=[10, 1240, 1156], 
            beta=0.99,
            gamma=2.0 # 👈 감마값을 2.0으로 상향 조정하여 어려운 클래스(0, 1)에 더 집중
        )
    )
)