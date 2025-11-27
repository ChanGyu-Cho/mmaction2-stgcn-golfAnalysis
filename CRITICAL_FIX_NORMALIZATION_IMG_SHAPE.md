# CRITICAL FIX: Normalization img_shape Mismatch

## Problem
ST-GCN 서버 추론이 50:50 예측을 출력 (로컬 99% 정확도와 달리)

## Root Cause Analysis

### Issue 1: Misleading Debug Log ✅ FIXED
- **Line 117**: `debug_log("normalize=none")` 출력하지만
- **Line 153**: 실제로는 `normalize_method='0to1'` 사용
- **결과**: 혼란스러운 로그 메시지

### Issue 2: Wrong img_shape for Cropped Videos ⚠️ CRITICAL
**서버 로그 분석:**
```
crop_bbox=(0, 0, 262, 508) provided but NOT used
keypoint preview: [0.0774115, 0.107124, ...]
```

**문제점:**
1. Crop된 영상에서 OpenPose 실행 → 키포인트 좌표는 **crop 기준** (0~262 width)
2. 기존 코드는 **전체 프레임 크기**(1920 width)로 정규화
3. 결과: `x/1920 = 150/1920 = 0.078` (너무 작은 값)
4. 학습 데이터는 **실제 영상 크기**로 정규화했음

**예시:**
```python
# Crop 영역: (0, 0, 262, 508)
# 코 위치 (crop 좌표): (130, 50)

# ❌ 기존 방식 (잘못됨):
img_shape = (1080, 1920)  # 전체 프레임
x_norm = 130 / 1920 = 0.068  # 너무 작음!

# ✅ 올바른 방식:
img_shape = (508, 262)  # crop 크기
x_norm = 130 / 262 = 0.496  # 중앙값
```

## Solution

### Modified File: `stgcn_tester.py`

**Line 114-127: Dynamic img_shape based on crop_bbox**
```python
# CRITICAL: Determine img_shape based on actual frame dimensions
# If crop_bbox is provided, keypoints are in CROPPED frame coordinates
# and must be normalized using CROP dimensions, NOT full frame!
# Training always used actual video frame size for normalization.
if crop_bbox is not None:
    x1, y1, x2, y2 = crop_bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    img_shape = (crop_height, crop_width)  # (H, W) format
    debug_log(f"Using crop dimensions for normalization: {img_shape} (crop_bbox={crop_bbox})")
else:
    img_shape = (1080, 1920)  # Default full frame
    debug_log(f"Using default full frame for normalization: {img_shape}")
```

**Removed:**
1. Line 117: Misleading `debug_log("normalize=none")` message
2. Lines 130-150: Config file img_shape override logic
3. Lines 213-217: Incorrect comment claiming crop_bbox is not used

## Expected Result

### Before Fix:
```
img_shape=(1080, 1920)
keypoint: [0.068, 0.046, ...]  # 너무 작음
pred_score: [0.5054, 0.4946]  # 랜덤 수준
```

### After Fix:
```
img_shape=(508, 262)  # crop 크기 사용
keypoint: [0.496, 0.098, ...]  # 정상 범위
pred_score: [0.0012, 0.9988]  # 99% 정확도
```

## Testing

1. **Docker 이미지 재빌드:**
```powershell
cd d:\mmaction2\docker
docker build -t your-image-name .
```

2. **서버 재시작 후 API 호출**

3. **로그 확인:**
```
Using crop dimensions for normalization: (508, 262) (crop_bbox=(0, 0, 262, 508))
csv_to_pkl: ... (normalize=0to1, img_shape=(508, 262))
Normalized keypoints to 0-1 range
keypoint preview: [0.4xx, 0.3xx, ...]  # 정상 범위 값
pred_score: tensor([0.00xx, 0.99xx])  # 99% 정확도
```

## Related Files
- `stgcn_tester.py`: Lines 114-135 (img_shape 계산)
- `utils.py`: Lines 97-108 (normalization 구현)
- `make_pkl.py`: Lines 183-189 (학습 데이터 정규화)

## Notes
- 학습 데이터는 각 영상의 **실제 크기**로 정규화됨
- Crop된 영상은 **crop 크기**를 사용해야 동일한 분포 유지
- Full frame 영상은 **(1080, 1920)** 기본값 사용
