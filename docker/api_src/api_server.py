from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import tempfile, base64, os
import logging
import sys
import hashlib
from typing import Optional, List

# test.py와 동일한 구조를 사용하는 stgcn_tester 모듈
from modules.stgcn_tester import run_stgcn_test
from modules.utils import debug_log

app = FastAPI(title="mmaction-stgcn-api")

# configure logging early so startup logs are visible
logging.basicConfig(level=logging.INFO)

class CSVBase64Request(BaseModel):
    csv_base64: str
    crop_bbox: Optional[List[int]] = None  # CRITICAL FIX: Optional (x, y, w, h) crop bounding box

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/mmaction_stgcn_test")
def mmaction_stgcn_test_endpoint(payload: CSVBase64Request):
    """
    ST-GCN 모델을 사용하여 CSV 데이터를 평가하고 결과를 반환합니다.
    test.py와 동일한 구조로 Runner.from_cfg() -> runner.test() 호출
    
    Payload:
        csv_base64: Base64-encoded CSV content
        crop_bbox: Optional [x, y, w, h] crop bounding box from YOLO detection
    """
    temp_csv = None
    try:
        if not payload.csv_base64:
            raise HTTPException(status_code=400, detail="csv_base64 is required")
        
        # CSV 디코딩 및 임시 파일 저장
        csv_text = base64.b64decode(payload.csv_base64).decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(csv_text)
            temp_csv = Path(tmp.name)
        
        # Log small debug summary about the received CSV so we can validate
        # that different requests produce different temp files and payloads.
        try:
            h = hashlib.sha1(csv_text.encode('utf-8')).hexdigest()
        except Exception:
            h = None
        debug_log(f"Received request, temp_csv={temp_csv}, sha1={h}, len={len(csv_text)}")
        try:
            debug_log(f"CSV preview: {csv_text[:200]!r}")
        except Exception:
            pass
        
        # CRITICAL FIX: Extract crop_bbox from payload for keypoint offset correction
        crop_bbox = None
        if payload.crop_bbox is not None:
            try:
                crop_bbox = tuple(payload.crop_bbox)
                debug_log(f"Crop bbox from payload: {crop_bbox}")
            except Exception:
                debug_log(f"Warning: crop_bbox conversion failed")
        
        # test.py와 동일한 구조로 테스트 실행 (with crop_bbox correction)
        result = run_stgcn_test(temp_csv, crop_bbox=crop_bbox)
        try:
            # short result summary for debugging
            if isinstance(result, dict):
                debug_log(f"Result summary: orig_pred_index={result.get('orig_pred_index')}, pred_index={result.get('pred_index')}, reduced_confidence={result.get('reduced_confidence')}")
        except Exception:
            pass

        # If the tester returned the parsed dict with a JSON-safe 'raw_dump',
        # include both the raw DumpResults list and the parsed summary so callers
        # (frontends) can access high-level fields like prediction/confidence
        # while preserving backward compatibility for callers that expect the
        # raw list under `result`.
        try:
            if isinstance(result, dict) and 'raw_dump' in result and isinstance(result['raw_dump'], list):
                return JSONResponse(status_code=200, content={
                    "message": "OK",
                    # legacy: keep raw DumpResults list for compatibility
                    "result": result['raw_dump'],
                    # new: include the parsed summary (prediction, probs, etc.)
                    "parsed": result,
                })
        except Exception:
            pass

        # fallback: return whatever run_stgcn_test produced
        return JSONResponse(status_code=200, content={
            "message": "OK",
            "result": result
        })
    except Exception as e:
        debug_log(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_csv and temp_csv.exists():
            try:
                os.remove(temp_csv)
            except Exception:
                pass

@app.on_event("startup")
def init_mmaction_registry():
    import sys, importlib, logging, traceback
    logging.info("Initializing mmaction registry...")
    if '/mmaction2' not in sys.path: sys.path.insert(0, '/mmaction2')
    try:
        import mmengine, mmcv, mmaction
        try:
            from mmaction.utils import register_all_modules
        except Exception:
            from mmaction.utils.setup_env import register_all_modules
        register_all_modules(init_default_scope=True)
        # Ensure mmaction's registry entries are visible in mmengine global registries
        try:
            import mmengine.registry as _me_reg
            mma_reg = importlib.import_module('mmaction.registry')
            # Sync several registries from mmaction into mmengine so that
            # datasets/transforms/models registered by mmaction are available
            # to mmengine (used by Runner and DATASETS.build).
            for reg_name in ('MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'EVALUATOR'):
                mma_reg_obj = getattr(mma_reg, reg_name, None)
                me_reg_obj = getattr(_me_reg, reg_name, None)
                if mma_reg_obj is None or me_reg_obj is None:
                    continue
                mma_dict = getattr(mma_reg_obj, 'module_dict', {}) or {}
                me_dict = getattr(me_reg_obj, 'module_dict', {}) or {}
                added = 0
                for name, cls in mma_dict.items():
                    if name not in me_dict:
                        try:
                            me_reg_obj.register_module(module=cls, name=name, force=True)
                            added += 1
                        except Exception:
                            logging.exception(f"Failed to register {name} into mmengine.{reg_name}")
                logging.info(f"Synchronized {added} mmaction entries into mmengine.{reg_name}")
        except Exception:
            logging.exception('Failed to synchronize mmaction registry into mmengine')
    except Exception:
        logging.exception("Failed to init mmaction registry")
    # Log environment and module locations for debugging
    try:
        logging.info("CWD=%s", os.getcwd())
        logging.info("PYTHONPATH sample=%s", sys.path[:5])
        try:
            import mmaction as _mma
            logging.info("mmaction module file=%s", getattr(_mma, '__file__', None))
        except Exception:
            logging.exception("mmaction import failed at startup logging")
    except Exception:
        logging.exception("Failed to write startup debug info")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 19031))
    uvicorn.run("api_src.api_server:app", host="0.0.0.0", port=port, reload=False)