# 🔍 ST-GCN 서버 추론 정확도 문제 - CSV 데이터 품질 검증

## 현재 상황
- ✅ PreNormalize2D 활성화 완료
- ✅ normalize='none' 적용 완료  
- ✅ img_shape=(508, 262) crop 크기 사용
- ❌ **여전히 50:50 예측** (`pred_score: [0.4981, 0.5019]`)

## 새로운 가설: CSV 데이터 품질 문제

### 의심되는 원인

#### 1. Interpolation 누락
**로컬 테스트** (`finetune_stgcn_test_single.py` Line 268):
```python
produced = openpose_crop.process_one_crop_dir(
    video_crop_dir,
    conf_thresh=0.0,
    interp_fill='zero',  # ✅ Interpolation 적용
    interp_limit=None,
    ...
)
```

**서버 (controller.py)**: 
- Interpolation 적용 여부 **불확실**
- conversation summary에 "Lines 489-520: Interpolation + tidy/wide DataFrame creation" 있지만 실제 코드 미확인

#### 2. Static Frame Trimming 차이
**로컬** (`make_pkl.py` Line 170-177):
```python
try:
    start_idx, end_idx = trim_static_frames(keypoint, fps=30)
    if start_idx > 0 or end_idx < (keypoint.shape[0] - 1):
        keypoint = keypoint[start_idx:(end_idx + 1)]
except Exception as e:
    print(f"[WARN] static trimming failed: {e}")
```

**서버** (`utils.py` Line 85-95):
```python
if trim_static_frames is not None:  # ← trim_static_frames 함수가 import 되었는지 불확실
    try:
        kp_2d = keypoint[0]
        start_idx, end_idx = trim_static_frames(kp_2d, fps=30)
        ...
```

#### 3. Missing Keypoints (0 값)
**서버 CSV 예시** (로그에서):
```
[148.63, 115.694, 153.355, 105.146, ...]
```
- 픽셀 좌표로 보임 (정상)
- 하지만 **confidence 0인 키포인트**가 많을 가능성
- Interpolation 없으면 **0 값이 많아서 모델 성능 저하**

#### 4. Confidence Threshold 차이
- 로컬: `conf_thresh=0.0` (모든 키포인트 유지 후 interpolation)
- 서버: Confidence threshold 적용 여부 불명

## 검증 방법

### 1. CSV 데이터 분석 스크립트 생성
**파일**: `debug_csv_analysis.py`

실행 방법:
```bash
# 서버에서 생성된 CSV 확인 (Docker 내부)
python /mmaction2/api_src/modules/debug_csv_analysis.py /mmaction2/api_src/modules/results/debug_csv_XXXXXXXX.csv

# 로컬 좋은 CSV와 비교
python debug_csv_analysis.py /path/to/local/good/skeleton.csv
```

분석 항목:
- ✅ Zero 값 비율 (X, Y, Confidence)
- ✅ 좌표 범위 (정규화 여부 확인)
- ✅ Frame-to-frame 움직임 (interpolation 확인)
- ✅ Static frames 비율 (trimming 필요 여부)
- ✅ 첫/마지막 프레임 움직임

### 2. stgcn_tester.py 수정
**Line 129-135**: 디버그 CSV 자동 저장
```python
debug_csv_path = repo_results_dir / f"debug_csv_{unique_id}.csv"
shutil.copy2(str(csv_path), str(debug_csv_path))
debug_log(f"DEBUG: Saved input CSV to {debug_csv_path}")
```

### 3. 예상 발견 사항

**만약 서버 CSV가 문제라면:**
```
📊 Zero Value Analysis:
  X coordinates with 0: 35.2%  ← ⚠️ 너무 높음!
  Y coordinates with 0: 38.7%  ← ⚠️ 너무 높음!
  Confidence with 0: 42.1%     ← ⚠️ Interpolation 누락

🔀 Frame-to-Frame Movement:
  X mean diff: 78.35 px, max diff: 245.12 px  ← ⚠️ 갑작스러운 점프 (interpolation 없음)
```

**로컬 CSV (정상):**
```
📊 Zero Value Analysis:
  X coordinates with 0: 2.3%   ← ✅ 정상
  Y coordinates with 0: 1.8%   ← ✅ 정상
  Confidence with 0: 0.0%      ← ✅ Interpolation 적용됨

🔀 Frame-to-Frame Movement:
  X mean diff: 3.24 px, max diff: 12.45 px  ← ✅ 부드러운 움직임
```

## 해결 방안

### Option A: controller.py에 interpolation 추가 확인
skeleton_metric-api의 controller.py가 실제로 interpolation을 적용하는지 확인:
```python
# 예상 위치: Lines 489-520
df_2d = interpolate_sequence(df_2d, conf_thresh=0.0, fill_method='zero')
```

### Option B: utils.csv_to_pkl에 interpolation 추가
서버 측 `utils.py`에 interpolation 로직 추가:
```python
# Line 85 이전에 추가
if interpolation_needed:
    from skeleton_interpolate import interpolate_sequence
    keypoint = interpolate_sequence(keypoint, conf_thresh=0.0, fill_method='zero')
```

### Option C: trim_static_frames import 확인
`utils.py`에서 `trim_static_frames`가 실제로 import 되었는지 확인:
```python
try:
    from skeleton_interpolate import trim_static_frames
except ImportError:
    trim_static_frames = None  # ← 이 경우 trimming 안 됨!
```

## 다음 단계

1. **즉시**: Docker 이미지 재빌드하여 debug CSV 저장 활성화
2. **API 호출** 후 생성된 `debug_csv_XXXXXXXX.csv` 분석
3. **비교**: 로컬 좋은 CSV vs 서버 문제 CSV
4. **수정**: Interpolation/Trimming 누락 시 controller.py 또는 utils.py 수정

## 관련 파일
- ✅ `stgcn_tester.py`: Lines 129-135 (debug CSV 저장 추가)
- ✅ `debug_csv_analysis.py`: 새로 생성됨
- ⏳ `controller.py`: skeleton_metric-api (워크스페이스 외부, 확인 필요)
- ⏳ `utils.py`: Lines 85-95 (trim_static_frames import 확인 필요)
