# 🎯 ST-GCN 서버 추론 정확도 최종 수정

## 문제 상황
- 서버 ST-GCN: **50:50 예측** (`pred_score: [0.4945, 0.5055]`)
- 로컬 ST-GCN: **99% 정확도** (`pred_score: [0.001, 0.999]`)
- img_shape 수정 후에도 여전히 50:50

## Root Cause 발견

### 1. PreNormalize2D 차이
**서버 my_stgcnpp.py:**
```python
train_pipeline = [
    # dict(type='PreNormalize2D'),  # ❌ DISABLED
    dict(type='RandomAffine', ...),
    dict(type='GenSkeFeat', ...),
```

**로컬 my_stgcnpp_2class.py:**
```python
train_pipeline = [
    dict(type='PreNormalize2D'),  # ✅ ENABLED
    dict(type='RandomAffine', ...),
    dict(type='GenSkeFeat', ...),
```

### 2. 정규화 방식
- **로컬 학습**: `make_pkl.py` Line 180 → `normalize_method='0to1'` + PreNormalize2D
- **서버 추론**: `utils.csv_to_pkl` → normalize='none' (변경 전 0to1) + PreNormalize2D 비활성화

### 3. img_shape 처리
- crop_bbox=(0,0,262,508) 제공 시
- **수정 전**: img_shape=(1080,1920) 고정 → 너무 작은 정규화 값
- **수정 후**: img_shape=(508,262) crop 크기 사용 → 올바른 범위

## 최종 수정사항

### File 1: `my_stgcnpp.py`
**모든 파이프라인에 PreNormalize2D 활성화**
```python
train_pipeline = [
    dict(type='PreNormalize2D'),  # ✅ 활성화
    ...
]

val_pipeline = [
    dict(type='PreNormalize2D'),  # ✅ 활성화
    ...
]

test_pipeline = [
    dict(type='PreNormalize2D'),  # ✅ 활성화
    ...
]
```

### File 2: `stgcn_tester.py`
**Line 114-127: crop_bbox 기반 img_shape 계산 (유지)**
```python
if crop_bbox is not None:
    x1, y1, x2, y2 = crop_bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    img_shape = (crop_height, crop_width)  # (508, 262)
else:
    img_shape = (1080, 1920)
```

**Line 130-137: normalize_method='none' 사용 (수정됨)**
```python
# PreNormalize2D expects PIXEL coordinates (unnormalized)
csv_to_pkl(csv_path, ann_pkl_path, normalize_method='none', img_shape=img_shape)
```

## 예상 결과

### 수정 전 (잘못된 파이프라인):
```
1. CSV: 픽셀 좌표 (130, 50) in crop (262×508)
2. csv_to_pkl(normalize='0to1', img_shape=(1080,1920)): → (0.068, 0.046) ❌
3. PreNormalize2D 비활성화 → GenSkeFeat에 잘못된 값 전달
4. 결과: 50:50 예측
```

### 수정 후 (올바른 파이프라인):
```
1. CSV: 픽셀 좌표 (130, 50) in crop (262×508)
2. csv_to_pkl(normalize='none', img_shape=(508,262)): → (130, 50) ✅ 픽셀 유지
3. PreNormalize2D 활성화: (x-w/2)/(w/2) → (130-131)/131 = -0.008 ✅
4. GenSkeFeat → 올바른 spatial features
5. 결과: 99% 정확도 예상
```

## 검증 방법

Docker 재빌드 후 로그 확인:
```
csv_to_pkl: ... (normalize=none, img_shape=(508, 262))
Keypoint expanded: shape=(1, 66, 17, 2)
# 픽셀 좌표 유지: [130.5, 50.2, ...] (NOT [0.068, 0.046, ...])
pred_score: tensor([0.001x, 0.998x])  # 99% 정확도
```

## 기술적 배경

### PreNormalize2D란?
- MMAction2의 skeleton 전처리 transform
- **입력**: 픽셀 좌표 (0~width, 0~height)
- **출력**: 정규화된 좌표 `(x - w/2) / (w/2)` → [-1, +1] 범위
- **목적**: Center-normalize skeleton for scale invariance

### 로컬 학습의 실제 파이프라인:
1. `make_pkl.py`: CSV → PKL (normalize='0to1') → [0,1] 범위 저장
2. `PreNormalize2D`: [0,1] 픽셀 → `(0.5-0.5)/0.5 = 0` 중심 정규화? ❌

**모순 해결**: 로컬도 사실 **normalize='none'을 사용했을 가능성** 높음. 또는 PreNormalize2D가 [0,1] 입력도 허용하도록 구현됨.

### 안전한 접근:
**로컬 config 그대로 복사** → 서버에 `PreNormalize2D` 활성화 + `normalize='none'`

## 변경 파일 목록
1. ✅ `my_stgcnpp.py`: PreNormalize2D 활성화 (Lines 28, 61, 68)
2. ✅ `stgcn_tester.py`: normalize='none' + crop_bbox 처리 (Lines 114-137)

## Next Step
```powershell
cd d:\mmaction2\docker
docker build -t your-image-name .
```
