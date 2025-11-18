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
ann_file = r"D:\golfDataset\crop_pkl\combined_3class.pkl"
test_ann_file = r"D:\golfDataset\crop_pkl\combined_3class_test.pkl"
EPOCH = MAX_EPOCHS
clip_len = 100

# ------------------------ Data Pipeline (Train) ------------------------
train_pipeline = [
    dict(type='PreNormalize2D'),
    # 좌표 공간 노이즈: 작은 가우시안 노이즈로 검출 불확실성에 강건하게 함
    dict(type='AddGaussianNoise', std=0.01, p=0.5),
    
    # 🚨 RandomShift 제거 (KeyError 발생으로 인한 조치)
    dict(
        type='RandomAffine',
        scale_range=(0.9, 1.1), # 스케일 변화 (보수적으로 완화)
        shift_range=(-0.05, 0.05), # 위치 변화 (보수적으로 완화)
        rotate_range=(-15, 15), # 회전 변화
        shear_range=(0, 0),
        p=0.5
    ),

    dict(type='GenSkeFeat', dataset='coco', feats=[FEATS]),
    # 일부 키포인트 드롭으로 누락 검출 시 견고성 확보
    dict(type='RandomKeypointDrop', drop_prob=0.05),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]
    ),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    # 프레임 인덱스 시간 왜곡: 샘플된 frame_inds에 소량의 시차를 추가
    dict(type='RandomTemporalJitter', max_shift=5, p=0.5),
    dict(type='PoseDecode'),
    # 프레임 드롭: 일부 프레임을 0으로 만들어 누락 시나리오에 강건하게 함
    dict(type='RandomFrameDrop', drop_prob=0.05),
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
    # GCN 입력은 heatmap 기반 GeneratePoseTarget을 사용하지 않음.
    # 좌우반전 TTA는 테스트 루프에서 모델에 원본/flip 두 입력을 주어 평균화하세요.
    dict(type='Collect', keys=('keypoint', 'label'), meta_keys=('frame_interval', 'label_index'))
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