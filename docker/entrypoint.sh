#!/bin/bash
set -euo pipefail

# Lightweight entrypoint for mmaction2 API image
# - ensures PYTHONPATH contains /mmaction2, waits for mounts, performs pre-import checks
# - attempts to import mmengine/mmcv/mmaction and run register_all_modules before starting the server
# - finally execs the provided command (or defaults to python -m uvicorn ...)

echo "[$(date --iso-8601=seconds)] entrypoint: starting"

# CRITICAL: Ensure threading variables are set for deterministic behavior
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# CRITICAL: Configure CUBLAS for deterministic behavior (required for torch deterministic mode)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Ensure /mmaction2 is on PYTHONPATH so imports inside the image find the repo
export PYTHONPATH="/mmaction2:/mmaction2/mmaction:/mmaction2/api_src:${PYTHONPATH:-}"
PYTHON=${PYTHON:-python}
echo "entrypoint: PYTHONPATH=${PYTHONPATH}"
echo "entrypoint: Threading vars set to 1 for deterministic behavior"
echo "entrypoint: CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"

WAIT_TIMEOUT=${WAIT_TIMEOUT:-60}
SLEEP_INTERVAL=${SLEEP_INTERVAL:-5}

wait_for_path() {
    path="$1"
    timeout="$2"
    elapsed=0
    while [ ! -e "$path" ]; do
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "[$(date --iso-8601=seconds)] entrypoint: timeout waiting for $path"
            return 1
        fi
        echo "[$(date --iso-8601=seconds)] entrypoint: waiting for $path (elapsed=${elapsed}s)"
        sleep "$SLEEP_INTERVAL"
        elapsed=$((elapsed + SLEEP_INTERVAL))
    done
    echo "[$(date --iso-8601=seconds)] entrypoint: found $path"
    return 0
}

# Wait for api_src to exist (helpful when mounting or copying into image)
wait_for_path "/mmaction2/api_src/api_server.py" "$WAIT_TIMEOUT" || true

# Python-based import & registry check with retries
preimport_check() {
    "$PYTHON" - <<'PY'
import sys, time, traceback

def try_imports():
    ok = True
    for name in ('mmengine','mmcv','mmaction'):
        try:
            __import__(name)
            print(f"import ok: {name}")
        except Exception:
            ok = False
            print(f"import failed: {name}")
            traceback.print_exc()
    if not ok:
        return False
    try:
        # try to register mmaction modules (may be in different locations)
        try:
            from mmaction.utils import register_all_modules
        except Exception:
            from mmaction.utils.setup_env import register_all_modules
        register_all_modules(init_default_scope=True)
        print('register_all_modules: ok')
        return True
    except Exception:
        print('register_all_modules: failed')
        traceback.print_exc()
        return False

RETRIES=6
SLEEP=5
for i in range(RETRIES):
    print(f'preimport check attempt {i+1}/{RETRIES} ...')
    if try_imports():
        sys.exit(0)
    time.sleep(SLEEP)
print('preimport checks failed after retries', file=sys.stderr)
sys.exit(2)
PY
}

echo "[$(date --iso-8601=seconds)] entrypoint: running preimport checks (this may take a while)"
if ! preimport_check; then
    echo "[$(date --iso-8601=seconds)] entrypoint: preimport checks failed"
    # print some sysinfo for debugging
    echo "-- ls /mmaction2 --"
    ls -la /mmaction2 || true
    echo "-- ls /mmaction2/api_src --"
    ls -la /mmaction2/api_src || true
    echo "[$(date --iso-8601=seconds)] entrypoint: exiting due to failed imports"
    exit 1
fi

echo "[$(date --iso-8601=seconds)] entrypoint: preimport checks passed"

# If invoked with arguments, exec them; otherwise start uvicorn using PORT env
if [ "$#" -gt 0 ]; then
    echo "[$(date --iso-8601=seconds)] entrypoint: exec user command: $@"
    # If the user command begins with 'python', replace it with the configured interpreter
    if [ "$1" = "python" ] || [ "$1" = "python3" ]; then
        shift 1
        exec "$PYTHON" "$@"
    else
        exec "$@"
    fi
else
    PORT=${PORT:-19031}
    echo "[$(date --iso-8601=seconds)] entrypoint: starting uvicorn on port $PORT"
    exec "$PYTHON" -m uvicorn api_src.api_server:app --host 0.0.0.0 --port "$PORT"
fi
