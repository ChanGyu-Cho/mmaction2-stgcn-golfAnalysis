# ----------------------------------------
# MMACTION + FastAPI Layer
# ----------------------------------------
FROM bocker060/mmaction2-base:latest


# 1. 의존성 파일을 먼저 복사하여 캐시 활용도를 높입니다.
# requirements.txt만 복사합니다. 이 파일이 변경되지 않으면 다음 pip install은 캐시를 사용합니다.
# Copy the merged requirements (base + API) and install them
# Copy the API-specific requirements (docker/requirements.txt). Do NOT copy the repo-root requirements.txt
# which pulls the full project dependencies and may conflict with the runtime environment.
COPY ./docker/requirements.txt /mmaction2/requirements.txt
# Ensure we install into the Conda Python that the base image uses at runtime
ENV PYTHON=/opt/conda/bin/python
RUN ${PYTHON} -m pip install --no-cache-dir -r /mmaction2/requirements.txt

# install py-spy for runtime profiling/debugging (이것도 자주 바뀌지 않으니 여기에 유지)
RUN ${PYTHON} -m pip install py-spy

# default workdir 유지
WORKDIR /mmaction2

ENV FORCE_CUDA="1"
ENV PORT=19031

# CRITICAL: Set threading environment variables to ensure deterministic behavior
# matching local environment (prevent multi-threading from causing non-deterministic results)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# CRITICAL: Configure CUBLAS for deterministic behavior
# Required when using torch.use_deterministic_algorithms(True) with CUDA >= 10.2
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Allow overriding PYTHONPATH at build/run time. Declare ARG so the variable is defined at build-time,
# then set ENV using that ARG to avoid BuildKit undefined-variable warnings.
ARG PYTHONPATH=""
ENV PYTHONPATH="/mmaction2:/mmaction2/mmaction:/mmaction2/api_src:${PYTHONPATH}"

# 3. 나머지 FastAPI 서버 코드 복사
# 이제 이 COPY 명령은 requirements.txt가 변경되지 않은 경우에도 이후의 레이어에만 영향을 미칩니다.
COPY ./docker/api_src /mmaction2/api_src/
COPY ./docker/api_src/model.pth /mmaction2/api_src/model.pth

# expose an extra test port (if needed)
EXPOSE 19031

WORKDIR /mmaction2/api_src

# Copy entrypoint that performs pre-import checks and starts the server
COPY ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Use entrypoint for initialization; CMD provides default args if no command provided
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "api_src.api_server:app", "--host", "0.0.0.0", "--port", "19031"]