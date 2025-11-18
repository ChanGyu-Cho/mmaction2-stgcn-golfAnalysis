# ===================================================================
# STGCN++ 5-Class Golf Action Recognition Config - FINAL, FINAL, FINAL, FINAL VERSION (STABLE)
# ===================================================================

# ------------------------ Base Configuration ------------------------
_base_ = '../../_base_/default_runtime.py'
default_scope = 'mmaction'

# ------------------------ Tunable hyperparameters ------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 0.001 
WEIGHT_DECAY = 0.0005 
MAX_EPOCHS = 50
PATIENCE = 10 
WARMUP_EPOCHS = 5 
FEATS='b' # 'j' = joint features (x, y)

# ------------------------ Path Configuration ------------------------
_load_checkpoint_path = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth"
dataset_type = 'PoseDataset'
ann_file = r"E:\golfDataset\dataset\crop_pkl\combined_3class.pkl"
test_ann_file = r"D:\golfDataset\dataset\crop_pkl\combined_3class_test.pkl"
EPOCH = MAX_EPOCHS
clip_len = 100

# ------------------------ Data Pipeline (Train) ------------------------
train_pipeline = [
    # ⭐️ PreNormalize2D: 키포인트 데이터를 (0, 0)을 중심으로 재배치 (상대 좌표 변환)
    dict(type='PreNormalize2D'), 
    # GenSkeFeat: x, y 좌표와 신뢰도(c)를 기반으로 GCN이 사용할 특징을 생성
    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    
    # ⭐️ Resize 제거: 이미 PKL 생성 시 정규화(0to1)를 수행하거나, GenSkeFeat가 처리할 수 있음.
    # ⭐️ RandomResizedCrop, RandomAffine 제거: 크롭 데이터로 인한 Assertion/ValueError 방지.
    
    dict(
        type='Flip', # 좌우 대칭 증강만 유지
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15], 
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]
    ),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    # FormatGCNInput: 최종적으로 (M, T, V, C) 텐서로 변환 (C는 FEATS에 따라 2 또는 3)
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

test_pipeline = val_pipeline 

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
        begin=0,
        end=WARMUP_EPOCHS,
        by_epoch=True,
        start_factor=0.1),
    dict(
        type='MultiStepLR',
        begin=WARMUP_EPOCHS,
        end=EPOCH,
        by_epoch=True,
        # use percentages of the configured EPOCH for milestones so they adapt
        # to changes in total epochs (e.g. for 50 epochs use ~30 and ~40)
        milestones=[int(EPOCH * 0.6), int(EPOCH * 0.8)],
        gamma=0.1
    )
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
        loss_cls=dict(
            type='CBFocalLoss', 
            loss_weight=1.0,
            # 🚨 클래스 인덱스(0, 1, 2, 3, 4) 순서에 따라 샘플 수를 정확히 반영
            samples_per_cls=[1362, 1156, 2431],
            beta=0.9999,
            gamma=2.0 
        )
    )
)