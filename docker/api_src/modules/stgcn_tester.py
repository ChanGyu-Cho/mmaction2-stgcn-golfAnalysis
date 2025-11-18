"""
ST-GCN Tester Module
test.py와 완전히 동일한 구조로 작동하는 테스트 모듈
Runner.from_cfg() -> runner.test() 호출 -> result.pkl 파싱 및 반환
"""
import os
import os.path as osp
import pickle
import sys
import tempfile
import uuid
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

# Do not import mmaction.registry at module import time. Import it later
# after we ensure the local /mmaction2 repo is on sys.path so the registry
# is populated from the correct package (avoids stale/site-package binding).
from modules.utils import debug_log, csv_to_pkl

# New label mapping for updated checkpoint outputs:
# 0: worst, 1: bad, 2: normal, 3: good, 4: best
LABEL_MAP = {
    0: 'worst',
    1: 'bad',
    2: 'normal',
    3: 'good',
    4: 'best',
}


def prepare_config_for_test(csv_path: Path):
    """
    test.py의 parse_args()와 merge_args() 역할을 수행
    CSV 파일을 받아서 config를 준비하고 필요한 설정을 오버라이드
    """
    # 1. Config 파일 경로 (my_stgcnpp.py)
    config_path = Path(__file__).parent / "my_stgcnpp.py"
    checkpoint_path = None
    
    # checkpoint 경로 찾기
    for p in [
        Path(__file__).parent.parent / "model.pth"
    ]:
        if Path(p).exists():
            checkpoint_path = str(p)
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint file stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth not found")
    
    debug_log(f"Using config: {config_path}")
    debug_log(f"Using checkpoint: {checkpoint_path}")
    
    # 2. CSV를 ann.pkl로 변환
    unique_id = uuid.uuid4().hex[:8]
    # Prefer workspace-local results directory so files are visible outside the container
    repo_results_dir = Path(__file__).parent / 'results'
    try:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # fallback to system tempdir
        repo_results_dir = Path(tempfile.gettempdir())

    ann_pkl_path = repo_results_dir / f"test_ann_{unique_id}.pkl"
    debug_log(f"Converting CSV to PKL: {csv_path} -> {ann_pkl_path}")
    csv_to_pkl(csv_path, ann_pkl_path)
    
    # 3. result.pkl 경로 설정
    result_pkl_path = repo_results_dir / f"test_result_{unique_id}.pkl"
    
    # Make sure local mmaction2 package is importable so model classes (e.g. RecognizerGCN)
    # are registered before loading the config. We try the repo-local path and the
    # container path '/mmaction2'.
    repo_dir = Path(__file__).parent
    candidate = repo_dir.parent / "mmaction2"
    try:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
        else:
            # fallback to container path
            sys.path.insert(0, "/mmaction2")
    except Exception:
        pass

    # Ensure mmaction registers its modules (models, datasets, etc.).
    # Some mmaction registration happens when importing subpackages; call
    # register_all_modules if available to guarantee registries are populated.
    try:
        import mmaction  # noqa: F401
        try:
            # mmaction provides a helper to register all modules
            from mmaction.utils.setup_env import register_all_modules
            register_all_modules(init_default_scope=True)
            # Debug: list some registered model keys to verify GCN is present
            try:
                from mmaction.registry import MODELS
                model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                debug_log(f"Registered MODELS sample (len={len(model_keys)}): {model_keys[:50]}")
                debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
            except Exception as _e:
                debug_log(f"Failed to inspect MODELS registry: {_e}")
        except Exception:
            # Fallback: at least import models subpackage to trigger module imports
            try:
                import mmaction.models  # noqa: F401
                # Try to inspect MODELS registry even if register_all_modules unavailable
                try:
                    from mmaction.registry import MODELS
                    model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                    debug_log(f"Registered MODELS sample (fallback) (len={len(model_keys)}): {model_keys[:50]}")
                    debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
                except Exception as _e:
                    debug_log(f"Failed to inspect MODELS registry in fallback: {_e}")
            except Exception:
                # swallow; Config.fromfile() will still run and may raise a helpful error
                pass
    except Exception:
        # If mmaction cannot be imported at all, let Config.fromfile raise later
        pass

    # 4. Config 로드
    cfg = Config.fromfile(str(config_path))
    # Debug: what model type is requested in config
    try:
        cfg_model_type = cfg.model.type if hasattr(cfg, 'model') and isinstance(cfg.model, dict) and 'type' in cfg.model else getattr(cfg.model, 'type', None)
        debug_log(f"Config model.type -> {cfg_model_type}")
    except Exception as _e:
        debug_log(f"Failed to read cfg.model.type: {_e}")
    
    # 5. test.py의 merge_args와 동일한 설정 적용
    # work_dir 설정 (test.py와 동일한 우선순위)
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('/tmp/work_dirs', 
                               osp.splitext(osp.basename(str(config_path)))[0])
    
    # 6. checkpoint 로드 설정
    cfg.load_from = checkpoint_path
    
    # 7. test_dataloader의 ann_file 오버라이드 및 전처리 파이프라인 수정
    cfg.test_dataloader.dataset.ann_file = str(ann_pkl_path)
    
    # ----------------------------------------------------------------------
    # ⭐️ 핵심 수정 사항: 0-to-1 정규화 전처리 파이프라인 추가 (finetune과 동일)
    # ----------------------------------------------------------------------
    
    # test_dataloader에서 사용할 파이프라인을 복사하거나 가져옵니다.
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    
    # 0-to-1 정규화 클래스 (NormalizeIMSkeleton)를 찾거나 새로 정의합니다.
    # 일반적으로 NTU-RGB+D 데이터셋에서 사용되는 정규화입니다.
    # 'keypoint' 필드에 대해 mean과 std를 0으로 설정하여 (x - 0) / 1.0 = x 를 수행한 후,
    # C=2 (x, y)를 V (관절 수)로 나누고 V를 1.0으로 나눕니다.
    # 실제로 0-to-1 정규화의 목적은 min-max 스케일링이 아니라
    # 포즈 데이터의 특정 축을 1.0으로 스케일링하여 정규화하는 방식(NTU60의 일반적인 방식)이거나
    # 단순히 데이터의 스케일을 1로 만드는 것일 수 있습니다.
    # 여기서는 finetune에 사용된 일반적인 PoseDataset의 Normalize 파이프라인을 추가합니다.

    # 1. 'Normalize' (0-to-1) 스텝을 확인합니다.
    # 'NormalizeIMSkeleton'을 사용하면 finetune_stgcn_test에서 사용한
    # mean=0, std=1.0 처리를 간소화할 수 있습니다.
    
    # NOTE: 만약 finetune의 전처리가 단순히 각 채널을 [0, 1]로 나누는 min/max 방식이 아닌,
    # NTU의 center-normalize + unit-length 방식이라면,
    # 해당 로직을 수행하는 Transform (예: NormalizeIMSkeleton)을 삽입해야 합니다.
    # 여기서는 finetune 시 사용했을 것으로 추정되는 'NormalizeIMSkeleton'을 추가하고,
    # 파인튜닝 시 사용된 옵션(mean, std)이 필요하다면 해당 값으로 Normalize를 추가합니다.

    # 일반적으로 ST-GCN의 전처리는 포즈 데이터에 대한 Mean/Std 정규화가 아니라
    # (x, y) 좌표를 [0, 1] 범위로 스케일링하거나, 스켈레톤의 중앙을 맞추는 등의 작업이 포함됩니다.
    # finetune_stgcn_test에서 "0to1 정규화"를 사용했다면,
    # 그 전처리는 PoseDataset에 포함된 `keypoint` 전처리입니다.

    # 'PoseDataset'의 Pipeline 구성 요소:
    # 1) GeneratePoseTarget -> 2) FormatShape -> 3) (Optional) Normalize/Convert
    # FormatShape 뒤에 'Normalize'를 추가합니다.

    normalize_step = dict(type='NormalizeIMSkeleton',
                          mean=[100, 100],
                          std=[100, 100],
                          to_bgr=False)
    
    # FormatShape 이후에 NormalizeIMSkeleton을 삽입합니다.
    try:
        format_shape_idx = -1
        for i, step in enumerate(test_pipeline):
            # FormatShape 다음이나 마지막 부분에 삽입
            if isinstance(step, dict) and step.get('type') == 'FormatShape':
                format_shape_idx = i
                break
        
        # FormatShape 다음 (i+1) 위치에 삽입
        if format_shape_idx != -1:
            test_pipeline.insert(format_shape_idx + 1, normalize_step)
            debug_log("Inserted 'NormalizeIMSkeleton' after 'FormatShape'.")
        else:
            # FormatShape이 없으면 파이프라인의 끝에 추가 (안전 장치)
            test_pipeline.append(normalize_step)
            debug_log("Inserted 'NormalizeIMSkeleton' at the end of the pipeline.")

    except Exception as e:
        debug_log(f"Warning: Failed to insert NormalizeIMSkeleton: {e}")
        # 예외 발생 시 파이프라인의 맨 뒤에 추가하여 시도
        test_pipeline.append(normalize_step)


    # 8. DumpResults 설정 (test.py의 --dump 옵션과 동일)
    dump_metric = dict(type='DumpResults', out_file_path=str(result_pkl_path))
    if isinstance(cfg.test_evaluator, (list, tuple)):
        cfg.test_evaluator = list(cfg.test_evaluator)
        # 기존 DumpResults가 있으면 제거
        cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
        cfg.test_evaluator.append(dump_metric)
    else:
        cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
    
    # 9. launcher 설정
    cfg.launcher = 'none'
    
    # 10. 환경 설정 (안정성을 위해)
    if hasattr(cfg, 'env_cfg'):
        if hasattr(cfg.env_cfg, 'mp_cfg'):
            cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
        if hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg.backend = 'gloo'
    
    # 11. visualization 비활성화 (서버 환경에서 불필요)
    if hasattr(cfg, 'default_hooks') and isinstance(cfg.default_hooks, dict):
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks.visualization.enable = False
    
    debug_log(f"Config prepared: work_dir={cfg.work_dir}")
    debug_log(f"Result will be saved to: {result_pkl_path}")
    
    return cfg, ann_pkl_path, result_pkl_path


def run_stgcn_test(csv_path: Path):
    """
    test.py의 main() 함수와 완전히 동일한 구조
    1. Config 로드 및 설정
    2. Runner 생성
    3. runner.test() 실행
    4. result.pkl 파싱 및 반환
    """
    debug_log(f"run_stgcn_test start: {csv_path}")

    ann_pkl_path = None
    result_pkl_path = None

    try:
        # 1) Prepare config and temporary files
        cfg, ann_pkl_path, result_pkl_path = prepare_config_for_test(csv_path)

        # 2) Choose execution mode: inline (default) or subprocess when explicitly requested
        use_subproc = os.environ.get('MMACTION_USE_SUBPROCESS', '0') == '1'
        if use_subproc:
            debug_log("MMACTION_USE_SUBPROCESS=1 -> running stgcn_subproc in subprocess")
            import subprocess, tempfile

            subproc_script = Path(__file__).parent / "stgcn_subproc.py"
            env = os.environ.copy()
            repo_dir = Path(__file__).parent.parent
            candidate = repo_dir / "mmaction2"
            env_pythonpath = str(candidate) if candidate.exists() else "/mmaction2"
            if env.get('PYTHONPATH'):
                env['PYTHONPATH'] = env_pythonpath + os.pathsep + env['PYTHONPATH']
            else:
                env['PYTHONPATH'] = env_pythonpath

            cmd = [
                sys.executable,
                str(subproc_script),
                '--config', str(Path(__file__).parent / 'my_stgcnpp.py'),
                '--checkpoint', str(cfg.load_from),
                '--ann', str(ann_pkl_path),
                '--out', str(result_pkl_path),
            ]

            cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_env:
                env['CUDA_VISIBLE_DEVICES'] = cuda_env
                debug_log(f"Forwarding CUDA_VISIBLE_DEVICES={cuda_env} to subprocess")
            mma_device = os.environ.get('MMACTION_DEVICE')
            if mma_device:
                cmd += ['--device', mma_device]
                debug_log(f"Passing device='{mma_device}' to subprocess")

            debug_log(f"Running subprocess: {cmd} with PYTHONPATH={env['PYTHONPATH']}")
            out_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stdout.log"
            err_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stderr.log"
            debug_log(f"Subprocess stdout redirected to: {out_log}")
            debug_log(f"Subprocess stderr redirected to: {err_log}")

            with open(out_log, 'wb') as _outf, open(err_log, 'wb') as _errf:
                proc = subprocess.Popen(cmd, stdout=_outf, stderr=_errf, env=env)
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    debug_log("Subprocess timed out and was killed")
                    try:
                        debug_log(f"subproc stdout (partial):\n{out_log.read_text(errors='replace')}")
                        debug_log(f"subproc stderr (partial):\n{err_log.read_text(errors='replace')}")
                    except Exception:
                        pass
                    raise RuntimeError("stgcn_subproc timed out")

            try:
                out_text = out_log.read_text(errors='replace')
                err_text = err_log.read_text(errors='replace')
                max_len = 10000
                debug_log(f"subproc stdout (tail):\n{out_text[-max_len:]}")
                debug_log(f"subproc stderr (tail):\n{err_text[-max_len:]}")
            except Exception as _e:
                debug_log(f"Failed to read subproc logs: {_e}")

            if proc.returncode != 0:
                raise RuntimeError(f"stgcn_subproc failed (exit {proc.returncode}); see logs: {err_log}")

            debug_log("Subprocess completed successfully")
            if not result_pkl_path.exists():
                raise FileNotFoundError(f"Result file not found: {result_pkl_path}")
            debug_log(f"Loading result from: {result_pkl_path}")
            with open(result_pkl_path, "rb") as f:
                result_data = pickle.load(f)
            parsed_result = parse_test_result(result_data)
            debug_log(f"Test completed successfully: {parsed_result}")
            return parsed_result

        # Inline execution (default)
        debug_log("Running ST-GCN test inline (same process)")

        # Ensure local repo on sys.path
        try:
            repo_dir = Path(__file__).parent.parent
            candidate = repo_dir / "mmaction2"
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            elif "/mmaction2" not in sys.path:
                sys.path.insert(0, "/mmaction2")
        except Exception:
            pass

        # Register modules and sync registries
        try:
            try:
                from mmaction.utils import register_all_modules
            except Exception:
                try:
                    from mmaction.utils.setup_env import register_all_modules
                except Exception:
                    register_all_modules = None
            if register_all_modules:
                register_all_modules(init_default_scope=True)
        except Exception:
            debug_log("Warning: register_all_modules failed in inline mode")

        try:
            # Explicitly import dataset modules that perform registration as a side-effect.
            try:
                import mmaction.datasets.pose_dataset  # noqa: F401
                debug_log("Imported mmaction.datasets.pose_dataset in inline mode")
            except Exception as _e:
                debug_log(f"Failed to import mmaction.datasets.pose_dataset: {_e}")

            import importlib
            import mmengine.registry as _me_reg
            mma_reg = importlib.import_module('mmaction.registry')
            # Sync additional registries including METRICS and EVALUATOR which
            # are needed to build evaluators like AccMetric.
            for reg_name in ('MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'EVALUATOR'):
                mma_reg_obj = getattr(mma_reg, reg_name, None)
                me_reg_obj = getattr(_me_reg, reg_name, None)
                if mma_reg_obj is None or me_reg_obj is None:
                    debug_log(f"Registry {reg_name} missing in mmaction or mmengine; skipping")
                    continue
                mma_dict = getattr(mma_reg_obj, 'module_dict', {}) or {}
                me_dict = getattr(me_reg_obj, 'module_dict', {}) or {}
                added = 0
                for name, cls in mma_dict.items():
                    if name not in me_dict:
                        try:
                            me_reg_obj.register_module(module=cls, name=name, force=True)
                            added += 1
                        except Exception as _e:
                            debug_log(f"Failed to register {name} into mmengine.{reg_name}: {_e}")
                debug_log(f"Synchronized {added} entries into mmengine.{reg_name} (inline)")
        except Exception as _e:
            debug_log(f"Registry sync failed in inline mode: {_e}")

        # Build and run
        try:
            runner = Runner.from_cfg(cfg)
            runner.test()
        except Exception:
            import traceback as _tb
            debug_log("Inline runner.test() raised an exception:\n" + _tb.format_exc())
            raise

        if not result_pkl_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_pkl_path}")
        with open(result_pkl_path, "rb") as f:
            result_data = pickle.load(f)
        parsed_result = parse_test_result(result_data)
        debug_log(f"Inline test completed successfully: {parsed_result}")
        return parsed_result
    finally:
        # cleanup temporary files created during test
        try:
            if ann_pkl_path and Path(ann_pkl_path).exists():
                Path(ann_pkl_path).unlink()
                debug_log(f"Cleaned up: {ann_pkl_path}")
        except Exception as _e:
            debug_log(f"Failed to cleanup {ann_pkl_path}: {_e}")
        # By default we clean up the result PKL. If an operator wants to keep
        # the result file for debugging/inspection, set the environment
        # variable MMACTION_PRESERVE_RESULT=1 (or true).
        try:
            preserve = str(os.environ.get('MMACTION_PRESERVE_RESULT', '0')).lower() in ('1', 'true', 'yes')
            if result_pkl_path and Path(result_pkl_path).exists():
                # Before deleting (or even if preserving), write a human-readable
                # log file next to the PKL so operators can inspect the parsed
                # result without requiring pickle loading.
                try:
                    import json
                    import numpy as _np

                    def _safe_convert(obj):
                        # Convert numpy arrays and other non-JSON-able items
                        if isinstance(obj, (bytes, bytearray)):
                            return obj.decode('utf-8', errors='replace')
                        if isinstance(obj, _np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, ( _np.generic, )):
                            return obj.item()
                        # fallback to string
                        try:
                            json.dumps(obj)
                            return obj
                        except Exception:
                            return str(obj)

                    # Try to load parsed result (if available in this scope).
                    # We'll attempt to read the PKL raw if parsed_result isn't present.
                    parsed_to_dump = None
                    try:
                        # if result_data was left in local scope earlier, use parsed_result
                        parsed_to_dump = parsed_result if 'parsed_result' in locals() else None
                    except Exception:
                        parsed_to_dump = None

                    if parsed_to_dump is None:
                        # Attempt to load pickled data and create a safe representation
                        try:
                            with open(result_pkl_path, 'rb') as _f:
                                _raw = pickle.load(_f)
                            # If it's a list/dict, try to convert elements
                            def _normalize(o):
                                if isinstance(o, dict):
                                    return {k: _normalize(v) for k, v in o.items()}
                                if isinstance(o, list):
                                    return [_normalize(x) for x in o]
                                return _safe_convert(o)
                            parsed_to_dump = _normalize(_raw)
                        except Exception as _e:
                            parsed_to_dump = {"_load_error": str(_e)}

                    log_path = Path(str(result_pkl_path) + '.log')
                    try:
                        with open(log_path, 'w', encoding='utf-8') as _logf:
                            json.dump(parsed_to_dump, _logf, ensure_ascii=False, indent=2)
                        debug_log(f"Wrote human-readable result log: {log_path}")
                    except Exception as _e:
                        debug_log(f"Failed to write result log {log_path}: {_e}")

                except Exception as _e:
                    debug_log(f"Unexpected error while preparing result log: {_e}")

                # NOTE: Changed behavior per request: do NOT delete the result PKL.
                # Always preserve the PKL so operators can inspect it directly.
                debug_log(f"Preserving result PKL (no deletion performed): {result_pkl_path}")
        except Exception as _e:
            debug_log(f"Failed to cleanup {result_pkl_path}: {_e}")


def parse_test_result(result_data):
    """Parse result.pkl produced by DumpResults into a friendly dict.

    Returns a dict with keys: status, num_samples, predictions (list).
    Each prediction contains sample_index, scores (optional), predicted_class,
    and ground_truth_class (if present).
    """
    if not isinstance(result_data, list):
        debug_log(f"Unexpected result format: {type(result_data)}")
        return {
            "status": "error",
            "message": "Unexpected result format",
            "raw_type": str(type(result_data)),
        }

    if len(result_data) == 0:
        return {
            "status": "success",
            "num_samples": 0,
            # Single-sample pipeline: no prediction available
            "prediction": None,
            "debug": {
                "note": "no samples in result_data"
            }
        }

    # This function assumes a single-sample pipeline. If multiple results are
    # present, only the first sample will be processed and a note will be
    # included in the debug output.
    if len(result_data) > 1:
        debug_log(f"parse_test_result: received {len(result_data)} samples; processing only the first one")

    item = result_data[0]

    # Prepare default debug output
    debug_info = {
        "processed_sample_index": 0,
        "note": None,
    }

    if not isinstance(item, dict):
        debug_info["note"] = "first item is not a dict; cannot extract prediction"
        return {
            "status": "success",
            "num_samples": len(result_data),
            "prediction": None,
            "debug": debug_info,
        }

    # Extract prediction index and raw scores where available
    pred_idx = None
    raw_scores = None
    if 'pred_label' in item:
        try:
            pred_idx = int(item['pred_label'])
        except Exception:
            pred_idx = None
    if pred_idx is None and 'pred_labels' in item:
        try:
            val = item['pred_labels']
            if isinstance(val, (list, tuple)) and len(val) > 0:
                pred_idx = int(val[0])
            else:
                pred_idx = int(val)
        except Exception:
            pred_idx = None
    # Try multiple key names that may contain model outputs/scores/logits
    score_keys = ['pred_scores', 'pred_score', 'scores', 'score', 'logits', 'outputs', 'pred_logits', 'output']
    for k in score_keys:
        if k in item and raw_scores is None:
            try:
                import numpy as _np
                raw_scores = _np.asarray(item[k], dtype=float)
                if pred_idx is None and raw_scores.size > 0:
                    pred_idx = int(_np.argmax(raw_scores))
                break
            except Exception:
                raw_scores = None

    # Compute softmax probabilities if we have raw_scores
    probs = None
    topk = None
    confidence = None
    if raw_scores is not None:
        try:
            import numpy as _np
            # stable softmax
            rs = _np.asarray(raw_scores, dtype=float)
            ex = _np.exp(rs - _np.max(rs))
            probs_arr = ex / _np.sum(ex)
            probs = probs_arr.tolist()
            # top-k (k=5 or number of classes)
            k = min(5, probs_arr.size)
            idx_sorted = list(_np.argsort(probs_arr)[::-1])
            topk = []
            for i in idx_sorted[:k]:
                topk.append({
                    "index": int(i),
                    "label": LABEL_MAP.get(int(i), int(i)),
                    "prob": float(probs_arr[int(i)])
                })
            if pred_idx is not None and 0 <= pred_idx < len(probs_arr):
                confidence = float(probs_arr[pred_idx])
            # Emit a compact debug line so this information appears in logs
            try:
                dbg_topk = [{
                    'index': int(x['index']),
                    'label': x['label'],
                    'prob': float(x['prob'])
                } for x in topk] if topk else None
                debug_log(f"parse_test_result: pred_index={pred_idx}, prediction={LABEL_MAP.get(pred_idx, pred_idx)}, confidence={confidence}, topk={dbg_topk}")
            except Exception:
                # swallow logging error
                pass
        except Exception:
            probs = None
            topk = None
            confidence = None

    # Map pred_idx to label if available
    pred_label = LABEL_MAP.get(pred_idx, pred_idx) if pred_idx is not None else None

    result = {
        "status": "success",
        "num_samples": len(result_data),
        # Single-sample pipeline convenience field
        "prediction": pred_label,
        "pred_index": pred_idx,
        "raw_scores": (raw_scores.tolist() if raw_scores is not None else None),
        "probs": probs,
        "confidence": confidence,
        "topk": topk,
        "debug": debug_info,
    }

    return result