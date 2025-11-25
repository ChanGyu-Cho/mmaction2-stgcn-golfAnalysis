#!/usr/bin/env python3
"""
Simplified ST-GCN tester (finetune-style)

This module provides a single entrypoint `run_stgcn_test(csv_path)` which:
 - converts a single CSV to an annotation PKL
 - prepares a MMAction2 test config and appends a `DumpResults` evaluator
 - runs `Runner.test()` inline while capturing raw model outputs (logits)
 - writes a `*.raw_full.pkl` containing DumpResults + annotations + raw_model_outputs
 - returns a JSON-friendly dict summarizing the prediction (probs/topk/etc.)

This file intentionally implements a focused, easy-to-audit flow (inspired by
`finetune_stgcn_test.py`) rather than the previous large, multi-mode tester.
"""

from pathlib import Path
import os
import sys
import uuid
import tempfile
import pickle
import json
from typing import Optional

import numpy as _np

# Ensure local mmaction2 in sys.path when running inside container/workspace
REPO_ROOT = Path(__file__).resolve().parents[1]
MMACTION2_CANDIDATE = REPO_ROOT / 'mmaction2'
if str(MMACTION2_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(MMACTION2_CANDIDATE))

from mmengine.config import Config
from mmengine.runner import Runner

try:
    # helper utilities live in modules.utils in this project
    from modules.utils import debug_log, csv_to_pkl
except Exception:
    # fallback trivial implementations
    def debug_log(*args, **kwargs):
        try:
            print(*args, **kwargs)
        except Exception:
            pass

    def csv_to_pkl(csv_path, out_pkl, normalize_method='none', img_shape=(1080, 1920)):
        # Very small helper: expect csv -> build simple ann.pkl with annotations list
        # This is intentionally minimal; the project-provided csv_to_pkl is preferred.
        import csv as _csv
        anns = []
        with open(csv_path, 'r', encoding='utf-8') as rf:
            r = _csv.DictReader(rf)
            for row in r:
                anns.append(row)
        obj = {'annotations': anns}
        with open(out_pkl, 'wb') as wf:
            pickle.dump(obj, wf, protocol=pickle.HIGHEST_PROTOCOL)


def _softmax(arr):
    try:
        a = _np.asarray(arr, dtype=float)
        a = a - _np.max(a)
        ex = _np.exp(a)
        s = ex.sum() if ex.sum() != 0 else 1.0
        return (ex / s).tolist()
    except Exception:
        try:
            import math
            arrf = [float(x) for x in arr]
            m = max(arrf)
            exps = [math.exp(x - m) for x in arrf]
            s = sum(exps) if sum(exps) != 0 else 1.0
            return [e / s for e in exps]
        except Exception:
            return None


def _ensure_repo_registration():
    """Try to register mmaction modules (best-effort)."""
    try:
        try:
            from mmaction.utils import register_all_modules
        except Exception:
            from mmaction.utils.setup_env import register_all_modules
        register_all_modules(init_default_scope=True)
        debug_log("Registered mmaction modules (best-effort)")
    except Exception as _e:
        debug_log(f"Warning: failed to register mmaction modules: {_e}")


def prepare_config_for_test(csv_path: Path) -> tuple:
    """Prepare a minimal test config and write ann/result PKL paths.

    Returns (cfg, ann_pkl_path, result_pkl_path)
    """
    repo_results_dir = Path(__file__).parent / 'results'
    try:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        repo_results_dir = Path(tempfile.gettempdir())

    unique_id = uuid.uuid4().hex[:8]
    ann_pkl_path = repo_results_dir / f"test_ann_{unique_id}.pkl"
    result_pkl_path = repo_results_dir / f"test_result_{unique_id}.pkl"

    # Convert CSV to PKL using project helper (preferred). Prefer to
    # normalize coordinates to 0..1 for ST-GCN inference.
    debug_log(f"Converting CSV to PKL: {csv_path} -> {ann_pkl_path} (normalize=0to1)")
    # Attempt to derive img_shape from cfg if present (later loaded), else use default
    img_shape = (1080, 1920)
    try:
        # try to peek at cfg file on disk to extract img_shape if available
        config_path = Path(__file__).parent / 'my_stgcnpp.py'
        if config_path.exists():
            try:
                # Load minimal config just to inspect dataset img_shape
                tmp_cfg = Config.fromfile(str(config_path))
                td = getattr(tmp_cfg, 'test_dataloader', None)
                if isinstance(td, dict):
                    ds = td.get('dataset', {})
                    if isinstance(ds, dict) and 'img_shape' in ds:
                        img_shape = tuple(ds['img_shape'])
                else:
                    try:
                        ds = getattr(td, 'dataset', None)
                        if ds is not None and getattr(ds, 'img_shape', None) is not None:
                            img_shape = tuple(ds.img_shape)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

    # csv_to_pkl in modules.utils expects Path-like objects so pass Paths (not str)
    try:
        csv_to_pkl(Path(csv_path), Path(ann_pkl_path), normalize_method='0to1', img_shape=img_shape)
    except TypeError:
        # fallback: if csv_to_pkl expects strings or differing signature, try string args
        try:
            csv_to_pkl(str(csv_path), str(ann_pkl_path), normalize_method='0to1', img_shape=img_shape)
        except TypeError:
            # ultimate fallback: call without normalization
            debug_log('csv_to_pkl signature incompatible; calling without normalize args')
            csv_to_pkl(Path(csv_path), Path(ann_pkl_path))

    # Load config file from same directory as this module (`my_stgcnpp.py`)
    config_path = Path(__file__).parent / 'my_stgcnpp.py'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = Config.fromfile(str(config_path))

    # Set ann_file and result dump
    try:
        if hasattr(cfg, 'test_dataloader'):
            try:
                cfg.test_dataloader.dataset.ann_file = str(ann_pkl_path)
            except Exception:
                try:
                    cfg.test_dataloader['dataset']['ann_file'] = str(ann_pkl_path)
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure a checkpoint is provided via cfg or environment
    checkpoint = os.environ.get('MMACTION_CHECKPOINT')
    if checkpoint:
        cfg.load_from = checkpoint

    # Append DumpResults evaluator
    dump_metric = dict(type='DumpResults', out_file_path=str(result_pkl_path))
    try:
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
    except Exception:
        cfg.test_evaluator = [dump_metric]

    # Inline/run settings
    cfg.launcher = 'none'
    try:
        # force deterministic single-sample inference
        if isinstance(cfg.test_dataloader, dict):
            cfg.test_dataloader['batch_size'] = 1
            cfg.test_dataloader['num_workers'] = 0
            cfg.test_dataloader['persistent_workers'] = False
        else:
            try:
                cfg.test_dataloader.batch_size = 1
                cfg.test_dataloader.num_workers = 0
                cfg.test_dataloader.persistent_workers = False
            except Exception:
                pass
    except Exception:
        pass

    # Ensure cfg.work_dir exists - Runner.from_cfg expects cfg['work_dir'] to be present
    try:
        if getattr(cfg, 'get', None) is not None:
            # mmengine.Config supports .get like dict-like access
            if cfg.get('work_dir', None) is None:
                cfg.work_dir = str(repo_results_dir / f"work_dir_{unique_id}")
        else:
            # fallback attribute access
            if not hasattr(cfg, 'work_dir') or cfg.work_dir in (None, ''):
                cfg.work_dir = str(repo_results_dir / f"work_dir_{unique_id}")
    except Exception:
        try:
            cfg.work_dir = str(repo_results_dir)
        except Exception:
            pass

    try:
        debug_log(f"Using cfg.work_dir={getattr(cfg, 'work_dir', None)}")
    except Exception:
        pass

    # Defensive: ensure returned paths are Path objects
    try:
        ann_pkl_path = Path(ann_pkl_path)
    except Exception:
        ann_pkl_path = Path(str(ann_pkl_path))
    try:
        result_pkl_path = Path(result_pkl_path)
    except Exception:
        result_pkl_path = Path(str(result_pkl_path))

    debug_log(f"Prepared config; result will be written to: {result_pkl_path}")
    return cfg, ann_pkl_path, result_pkl_path


def _to_numpy_safe(x):
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        if isinstance(x, _np.ndarray):
            return x
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        try:
            return _np.asarray(x)
        except Exception:
            return None
    return None


def _format_row(arr, max_items=100):
    """Return a short string preview of numeric array `arr` for logging.

    Truncates long arrays and formats floats with limited precision.
    """
    try:
        a = _to_numpy_safe(arr)
        if a is None:
            return repr(arr)
        # flatten
        try:
            flat = a.flatten()
        except Exception:
            flat = _np.asarray(a)
        L = int(getattr(flat, 'size', len(flat) if hasattr(flat, '__len__') else 0))
        preview = []
        for i, v in enumerate(flat.tolist() if hasattr(flat, 'tolist') else list(flat)):
            if i >= max_items:
                break
            try:
                preview.append(f"{float(v):.6g}")
            except Exception:
                preview.append(repr(v))
        tail = '...' if L > max_items else ''
        return f"len={L} [{', '.join(preview)}]{tail}"
    except Exception:
        try:
            return repr(arr)
        except Exception:
            return '<unprintable>'


def _to_json_safe(obj):
    """Recursively convert objects into JSON-serializable Python types.

    - torch.Tensor -> list
    - numpy.ndarray -> list
    - numpy scalars -> Python scalars
    - Path -> str
    - dict/list/tuple -> recursively converted
    - other unknown objects -> repr(obj)
    """
    try:
        # avoid heavy imports at module import time
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            try:
                return obj.detach().cpu().tolist()
            except Exception:
                return None
    except Exception:
        pass

    try:
        import numpy as _npy
        if isinstance(obj, _npy.ndarray):
            try:
                return obj.tolist()
            except Exception:
                # fall through
                pass
        if isinstance(obj, (_npy.generic,)):
            try:
                return obj.item()
            except Exception:
                pass
    except Exception:
        pass

    # basic types
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # containers
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = k if isinstance(k, str) else str(k)
            except Exception:
                key = repr(k)
            out[key] = _to_json_safe(v)
        return out

    # Path -> str
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:
        pass

    # numpy scalar handled above; for other scalars try to cast
    try:
        # attempt to extract .item() if present
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            return obj.item()
    except Exception:
        pass

    # bytes
    try:
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except Exception:
                return repr(obj)
    except Exception:
        pass

    # Fallback: string representation
    try:
        return repr(obj)
    except Exception:
        return None


def _find_logits_from_raw(raw):
    # Try to find a numeric vector in captured raw outputs
    if raw is None:
        return None
    candidate = None
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        for e in raw:
            if e is not None:
                candidate = e
                break
    else:
        candidate = raw

    if candidate is None:
        return None

    # If it's an ndarray-like
    try:
        if isinstance(candidate, _np.ndarray):
            return candidate
    except Exception:
        pass

    # If it's a dict, check common keys
    if isinstance(candidate, dict):
        for key in ('logits', 'outputs', 'pred_scores', 'scores', 'probs'):
            if key in candidate:
                arr = _to_numpy_safe(candidate[key])
                if arr is not None and arr.size >= 2:
                    return arr
        # fallback: check values
        for v in candidate.values():
            arr = _to_numpy_safe(v)
            if arr is not None and getattr(arr, 'size', 0) >= 2:
                return arr

    # If it's list/tuple, try first array-like
    if isinstance(candidate, (list, tuple)):
        for v in candidate:
            arr = _to_numpy_safe(v)
            if arr is not None and getattr(arr, 'size', 0) >= 2:
                return arr
        arr = _to_numpy_safe(candidate)
        if arr is not None and getattr(arr, 'size', 0) >= 2:
            return arr

    return None


def parse_result(result_data, raw_model_outputs: Optional[list]):
    """Produce a JSON-friendly summary from DumpResults + raw outputs.

    Returns a dict with fields: status, orig_pred_index, raw_probs, probs, topk
    """
    try:
        # normalize result_data
        if not isinstance(result_data, list):
            return {"status": "error", "message": "unexpected result format"}

        first = result_data[0] if result_data else None
        raw_probs = None
        probs = None
        orig_idx = None

        # prefer raw_model_outputs if available
        if raw_model_outputs:
            cand = _find_logits_from_raw(raw_model_outputs)
            if cand is not None:
                try:
                    arr = cand.flatten()
                except Exception:
                    arr = cand
                # Preserve raw numeric outputs for debugging; do NOT auto-apply softmax here.
                try:
                    raw_probs = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                except Exception:
                    raw_probs = None
                # log raw row preview for debugging
                try:
                    debug_log(f"RAW ROW from model outputs: {_format_row(arr, max_items=200)}")
                except Exception:
                    pass
                # Do not compute `probs` or derive orig_idx via argmax automatically
                # as that forces one-hot/int-like behavior which hides raw values.

        # fallback: inspect DumpResults first item
        if probs is None and isinstance(first, dict):
            for k in ('pred_scores', 'pred_score', 'scores', 'score', 'probs', 'logits', 'outputs'):
                if k in first:
                    v = first.get(k)
                    arr = _to_numpy_safe(v)
                    if arr is not None and getattr(arr, 'size', 0) > 0:
                        raw_probs = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                        # if values sum ~1, treat as probs
                        try:
                            s = float(_np.sum(arr))
                        except Exception:
                            s = None
                        if s is not None and abs(s - 1.0) < 1e-3:
                            probs = [float(x) for x in arr.tolist()]
                            try:
                                debug_log(f"RAW ROW from DumpResults (sum~1): {_format_row(arr, max_items=200)}")
                            except Exception:
                                pass
                        else:
                            # raw numeric values present in DumpResults; log them for debugging
                            try:
                                debug_log(f"RAW ROW from DumpResults (not normalized): {_format_row(arr, max_items=200)}")
                            except Exception:
                                pass
                        # Do NOT auto-apply softmax for logits/outputs; keep raw values
                        break

        # compute topk
        topk = None
        confidence = None
        if probs is not None:
            try:
                idxs = list(_np.argsort(_np.asarray(probs))[::-1])
                topk = [{"index": int(i), "prob": float(probs[int(i)])} for i in idxs[:min(5, len(idxs))]]
                confidence = float(probs[int(idxs[0])]) if idxs else None
            except Exception:
                topk = None

        # orig_pred_index try to fill from DumpResults
        if orig_idx is None and isinstance(first, dict):
            if 'pred_label' in first:
                try:
                    orig_idx = int(first.get('pred_label'))
                except Exception:
                    orig_idx = None
            elif 'pred_labels' in first:
                try:
                    pl = first.get('pred_labels')
                    if isinstance(pl, (list, tuple)) and pl:
                        orig_idx = int(pl[0])
                    else:
                        orig_idx = int(pl)
                except Exception:
                    orig_idx = None

        # Determine whether model already outputs 2-class probabilities/logits
        pred_index = None
        reduced_confidence = None
        binary_probs = None
        try:
            # Case A: probs explicitly present and length==2 -> treat as binary outputs
            if probs is not None and isinstance(probs, (list, tuple)) and len(probs) == 2:
                binary_probs = [float(probs[0]), float(probs[1])]
                # argmax -> 0 or 1
                pred_index = int(_np.argmax(_np.asarray(binary_probs)))
                prediction = "Professional" if pred_index == 1 else "Ordinary"
                reduced_confidence = float(binary_probs[pred_index])

            # Case B: raw_model_outputs contain a vector of length 2 (logits) and probs not set
            elif raw_model_outputs and isinstance(raw_model_outputs, (list, tuple)):
                cand = _find_logits_from_raw(raw_model_outputs)
                if cand is not None:
                    arr = _to_numpy_safe(cand)
                    if arr is not None and getattr(arr, 'size', 0) == 2:
                        # raw logits -> softmax to get probs
                        s = _softmax(arr)
                        if s is not None:
                            binary_probs = [float(s[0]), float(s[1])]
                            pred_index = int(_np.argmax(_np.asarray(binary_probs)))
                            prediction = "Professional" if pred_index == 1 else "Ordinary"
                            reduced_confidence = float(binary_probs[pred_index])

            # Case C: fallback for 5-class outputs (old behavior)
            elif probs is not None and isinstance(probs, (list, tuple)) and len(probs) >= 5:
                prob3 = float(probs[3])
                prob4 = float(probs[4])
                prob0 = float(probs[0])
                prob1 = float(probs[1])
                prob2 = float(probs[2])
                binary_probs = [float(prob0 + prob1 + prob2), float(prob3 + prob4)]

                prof_max = max(prob3, prob4)
                ord_max = max(prob0, prob1, prob2)
                # threshold decision
                if prof_max >= 0.5:
                    pred_index = 1
                    prediction = "Professional"
                    reduced_confidence = float(prof_max)
                elif ord_max >= 0.5:
                    pred_index = 0
                    prediction = "Ordinary"
                    reduced_confidence = float(ord_max)
                else:
                    if orig_idx is not None:
                        pred_index = 0 if orig_idx in (0, 1, 2) else 1
                        prediction = "Ordinary" if pred_index == 0 else "Professional"
                    else:
                        pred_index = None
                        prediction = None

            else:
                # No probs available; if orig_idx present and already binary (0/1) respect it
                if orig_idx is not None:
                    # If orig_idx looks binary, use it directly; else map 5->2
                    try:
                        oi = int(orig_idx)
                        if oi in (0, 1):
                            pred_index = oi
                            prediction = "Professional" if pred_index == 1 else "Ordinary"
                        else:
                            pred_index = 0 if oi in (0, 1, 2) else 1
                            prediction = "Ordinary" if pred_index == 0 else "Professional"
                    except Exception:
                        pred_index = None
                        prediction = None
                else:
                    pred_index = None
                    prediction = None
        except Exception:
            pred_index = None
            reduced_confidence = None
            binary_probs = None
            prediction = None

        # Ensure the raw DumpResults are JSON-serializable for API responses
        try:
            safe_raw_dump = _to_json_safe(result_data)
        except Exception:
            try:
                safe_raw_dump = repr(result_data)
            except Exception:
                safe_raw_dump = None

        return {
            "status": "success",
            "orig_pred_index": orig_idx,
            "pred_index": pred_index,
            "prediction": prediction,
            "raw_probs": _to_json_safe(raw_probs) if raw_probs is not None else None,
            "probs": _to_json_safe(probs) if probs is not None else None,
            "binary_probs": _to_json_safe(binary_probs) if binary_probs is not None else None,
            "confidence": _to_json_safe(confidence) if confidence is not None else None,
            "reduced_confidence": _to_json_safe(reduced_confidence) if reduced_confidence is not None else None,
            "topk": _to_json_safe(topk) if topk is not None else None,
            "raw_dump": safe_raw_dump,
        }
    except Exception as _e:
        debug_log(f"parse_result error: {_e}")
        return {"status": "error", "message": str(_e)}


def run_stgcn_test(csv_path: str):
    """Run ST-GCN on a single CSV file and return parsed result dict.

    This is the primary external entrypoint expected by the API server.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    cfg, ann_pkl, result_pkl = prepare_config_for_test(csv_path)
    # Ensure we are working with Path objects (prepare_config_for_test should return Paths,
    # but coerce defensively in case callers or prior code returned strings).
    try:
        ann_pkl = Path(ann_pkl)
    except Exception:
        ann_pkl = Path(str(ann_pkl))
    try:
        result_pkl = Path(result_pkl)
    except Exception:
        result_pkl = Path(str(result_pkl))

    # Debug: log input PKL keypoint arrays (shape + short preview) before inference
    try:
        with open(ann_pkl, 'rb') as af:
            _ann = pickle.load(af)
        _anns = _ann.get('annotations') if isinstance(_ann, dict) else None
        if _anns:
            for idx, a in enumerate(_anns):
                try:
                    kp = a.get('keypoint')
                    # convert to numpy safely for shape/preview
                    try:
                        kp_arr = _np.asarray(kp)
                    except Exception:
                        kp_arr = kp
                    try:
                        shp = getattr(kp_arr, 'shape', None)
                    except Exception:
                        shp = None
                    debug_log(f"ANN PKL[{idx}] frame_dir={a.get('frame_dir')} keypoint.shape={shp}")
                    # preview first person's first frame (or flatten if small)
                    try:
                        preview_src = kp_arr
                        if hasattr(kp_arr, 'ndim') and kp_arr.ndim >= 3:
                            preview_src = kp_arr[0, 0]
                        debug_log(f"ANN PKL[{idx}] keypoint preview: {_format_row(preview_src, max_items=200)}")
                    except Exception as _e:
                        debug_log(f"ANN PKL[{idx}] preview error: {_e}")
                except Exception as _e:
                    debug_log(f"Failed to log annotation[{idx}] coords: {_e}")
    except Exception as _e:
        debug_log(f"Could not open/parse ann_pkl for logging: {_e}")

    # Best-effort: register mmaction modules
    _ensure_repo_registration()

    # Create runner and instrument model to capture raw outputs
    runner = Runner.from_cfg(cfg)

    _raw_outputs = []
    try:
        model = getattr(runner, 'model', None) or getattr(runner, 'module', None)
        orig_forward = getattr(model, 'forward', None) if model is not None else None
        if orig_forward is not None:
            def _to_cpu(x):
                try:
                    import torch as _torch
                    if isinstance(x, _torch.Tensor):
                        return x.detach().cpu().numpy()
                except Exception:
                    pass
                if isinstance(x, (list, tuple)):
                    return [_to_cpu(v) for v in x]
                if isinstance(x, dict):
                    return {k: _to_cpu(v) for k, v in x.items()}
                return x

            def _wrapped_forward(*a, **kw):
                out = orig_forward(*a, **kw)
                try:
                    _raw_outputs.append(_to_cpu(out))
                except Exception:
                    pass
                return out

            try:
                model.forward = _wrapped_forward
                debug_log("Wrapped model.forward to capture raw outputs")
            except Exception as _e:
                debug_log(f"Failed to wrap model.forward: {_e}")

    except Exception as _e:
        debug_log(f"Instrumentation failed: {_e}")

    # Run test (may write result_pkl via DumpResults)
    try:
        runner.test()
    except Exception as _e:
        debug_log(f"runner.test failed: {_e}")
        raise

    # Load DumpResults
    if not result_pkl.exists():
        raise FileNotFoundError(f"Result PKL not found: {result_pkl}")

    with open(result_pkl, 'rb') as f:
        result_data = pickle.load(f)

    # Persist raw_full.pkl next to result for debugging
    try:
        anns = None
        try:
            with open(ann_pkl, 'rb') as af:
                _ann = pickle.load(af)
                anns = _ann.get('annotations') if isinstance(_ann, dict) else None
        except Exception:
            anns = None

        full_obj = {'dump_results': result_data, 'annotations': anns}
        if _raw_outputs:
            full_obj['raw_model_outputs'] = _raw_outputs
        # Log the incoming result_pkl type/value for debugging (short-circuit noisy dumps)
        try:
            debug_log(f"Persisting full PKL; result_pkl type={type(result_pkl)}, repr={repr(result_pkl)[:200]}")
        except Exception:
            pass

        # Defensive: ensure result_pkl is a Path so `.stem` is available
        try:
            rp = Path(result_pkl)
        except Exception:
            try:
                rp = Path(str(result_pkl))
            except Exception:
                # fallback: use repo_results_dir + a unique name
                rp = Path(__file__).parent / 'results' / f"test_result_fallback_{uuid.uuid4().hex[:8]}.pkl"
                debug_log(f"Warning: could not coerce result_pkl; using fallback {rp}")

        full_pkl = rp.with_name(rp.stem + '.raw_full.pkl')
        try:
            with open(full_pkl, 'wb') as ff:
                pickle.dump(full_obj, ff, protocol=pickle.HIGHEST_PROTOCOL)
            debug_log(f"Wrote full raw PKL: {full_pkl}")
        except Exception as _e:
            debug_log(f"Failed to write full raw PKL: {_e}")
    except Exception:
        pass

    # Parse and return result
    parsed = parse_result(result_data, raw_model_outputs=_raw_outputs if _raw_outputs else None)
    # Additional logging: if we captured raw outputs list, log a short preview
    try:
        if _raw_outputs:
            try:
                preview = _format_row(_raw_outputs[0], max_items=200)
            except Exception:
                preview = repr(_raw_outputs[0])
            debug_log(f"Captured raw_model_outputs[0] preview: {preview}")
    except Exception:
        pass
    return parsed


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='CSV file to run inference on')
    args = p.parse_args()
    res = run_stgcn_test(args.csv)
    print(json.dumps(res, indent=2, ensure_ascii=False))
