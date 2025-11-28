# Docker 환경 동기화 분석 및 수정 사항

## 분석 날짜
2025-11-28

## 문제 상황
- 로컬에서 생성한 skeleton CSV는 0 값이 없고 높은 정확도로 추론
- 서버(Docker)에서 생성한 skeleton CSV는 0 값이 산재하고 추론 정확도가 50:50으로 낮음
- 동일한 모델과 전처리 코드를 사용함에도 서버에서 추론이 제대로 작동하지 않음

## 발견된 환경 차이

### 1. Python 버전 차이 (수정하지 않음)
- **로컬**: Python 3.8.20
- **Docker**: Python 3.10+ (pytorch/pytorch base image 기본값)
- **영향**: 부동소수점 연산의 미묘한 차이 가능
- **결정**: Python 다운그레이드 시 conda 의존성 충돌 발생. PyTorch 베이스 이미지의 Python 3.10 유지.
  Python 버전 차이보다 **NumPy와 OpenCV 버전이 훨씬 더 중요**하므로 이 두 라이브러리만 정확히 일치시킴.

### 2. NumPy 버전 차이
- **로컬**: NumPy 1.24.3 (정확히 고정)
- **Docker**: NumPy <2 (범위만 지정, 실제 버전 불명)
- **영향**: 배열 연산 및 부동소수점 정밀도에 직접적 영향

### 3. OpenCV 버전 차이 ⚠️ **가장 중요**
- **로컬**: OpenCV 4.12.0
- **Docker**: OpenCV >=4.6.0,<4.12 (4.12 제외됨)
- **영향**: 
  - OpenCV 4.12는 NumPy 2.x 호환성을 위한 대규모 변경 포함
  - 이미지 보간, 크로핑, 리사이징 알고리즘 차이
  - OpenPose 후처리에서 skeleton 좌표 계산 차이 발생 가능

### 4. 멀티스레딩 환경변수 미설정
- **문제**: OMP_NUM_THREADS, MKL_NUM_THREADS 등 미설정
- **영향**: 비결정적 수치 연산 발생 가능

### 5. PyTorch 결정론적 설정 누락
- **문제**: cudnn.deterministic, cudnn.benchmark 설정 없음
- **영향**: CUDA 연산의 비결정성

## 적용된 수정 사항

### 1. Dockerfile.base 수정
```dockerfile
# Python 버전: PyTorch 베이스 이미지의 3.10 유지 (다운그레이드 시 conda 충돌)
# 대신 NumPy와 OpenCV 버전을 정확히 일치시킴

# NumPy 버전을 정확히 고정
RUN ${PYTHON} -m pip install --no-cache-dir "numpy==1.24.3"
```

### 2. requirements.txt 수정
```python
# NumPy 정확한 버전 고정
numpy==1.24.3

# OpenCV 버전을 로컬과 일치시킴
opencv-python==4.12.0
```

### 3. Dockerfile 환경변수 추가
```dockerfile
# 멀티스레딩 제어로 결정론적 연산 보장
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
```

### 4. entrypoint.sh 환경변수 추가
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 5. api_server.py PyTorch 설정 추가
```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### 6. stgcn_tester.py PyTorch 설정 추가
동일한 결정론적 설정을 모듈 초기화 시점에 적용

## 예상 효과

1. **OpenCV 버전 일치**: skeleton 좌표 계산의 정확한 재현
2. **NumPy 버전 고정**: 정규화 및 배열 연산의 일관성
3. **Python 버전 일치**: 부동소수점 연산의 정확한 재현
4. **결정론적 설정**: 추론 결과의 재현성 보장

## 다음 단계

1. Docker 이미지 재빌드:
   ```bash
   cd d:\mmaction2
   docker build -f docker\Dockerfile.base -t mmaction2-base .
   docker build -f docker\Dockerfile -t mmaction2-api .
   ```

2. 컨테이너 재시작 후 동일한 CSV로 테스트

3. 로컬과 서버의 skeleton CSV 및 추론 결과 비교

## 추가 확인 사항

만약 위 수정 후에도 차이가 발생한다면:

1. **OpenPose 버전 차이**: 서버와 로컬의 OpenPose 바이너리 버전 확인
2. **YOLO 크롭 차이**: 크롭 좌표 계산 방식의 차이
3. **보간 강도**: skeleton 보간 알고리즘 파라미터 차이
4. **프레임 트리밍**: static frame 제거 로직의 차이
