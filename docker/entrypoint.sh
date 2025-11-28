#!/bin/sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MMACTION_CHECKPOINT="/mmaction2/api_src/model.pth"
export PYTHONPATH="/mmaction2:/mmaction2/mmaction:/mmaction2/api_src:${PYTHONPATH:-}"
PYTHON=${PYTHON:-python}
echo "entrypoint: starting"
echo "MMACTION_CHECKPOINT=${MMACTION_CHECKPOINT}"
$PYTHON -m uvicorn api_src.api_server:app --host 0.0.0.0 --port 19031
