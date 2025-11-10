_base_ = '../../_base_/default_runtime.py'

# Do not use global `load_from` so that the full checkpoint (including
# classifier head) is not automatically loaded. Instead we explicitly
# initialize only the backbone from the pretrained checkpoint below.
_load_checkpoint_path = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"

dataset_type = 'PoseDataset'
ann_file = r"E:\golfDataset\dataset\crop_pkl\combined_5class.pkl"
test_ann_file = r"D:\golfDataset\dataset\crop_pkl\skeleton_dataset_test.pkl"

# Runtime settings (safe defaults for API/container)
EPOCH = 50
clip_len = 50
fp16 = None
auto_scale_lr = dict(enable=False, base_batch_size=128)

# ===================================================================
# ⭐️ [최종 수정] 데이터 증강(Data Augmentation) - Flip 사용
# ===================================================================
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    # --- V4 수정: 'Flip'을 사용하고 스켈레톤 매핑 정보를 전달합니다. ---
    dict(
        type='Flip',
        flip_ratio=0.5,
        # 이전에 사용하던 키포인트 매핑 정보를 다시 전달하여 스켈레톤 Flip이 정확히 이루어지도록 합니다.
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15], 
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]
    ),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=clip_len, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
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
    batch_size=16,
    num_workers=2,
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
    num_workers=2,
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

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=EPOCH, # 50 Epoch
        by_epoch=True,
        # LR이 너무 일찍 떨어지는 것을 방지 (예: 50 Epoch 중 30, 40 Epoch에서 감소)
        milestones=[30, 40], 
        gamma=0.1
    )
]

# ===================================================================
# ⭐️ [과적합 방지] 학습률 및 정규화(재확인)
# ===================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.001, 
        momentum=0.9,
        weight_decay=0.005, 
        nesterov=True),
    # 경사 클리핑을 5에서 2로 강화하여 불안정성 해소
    clip_grad=dict(max_norm=0.5, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=128)

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
        num_classes=5,
        in_channels=256,
        loss_cls=dict(
            type='CrossEntropyLoss', 
            loss_weight=1.0,
            # ⭐️ [필수 수정] 클래스 가중치 추가
            # 인덱스 0부터 순서대로 클래스 0, 1, 2, 3, 4에 대한 가중치
            class_weight=[3.0, 0.4, 0.4, 0.6, 1.2] 
        )
    )
)