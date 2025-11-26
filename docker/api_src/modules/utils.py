import datetime, os, threading, sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
try:
    # try to import trim_static_frames used by other project scripts
    try:
        # project-specific module located in other folders (best-effort)
        from module import trim_static_frames
    except Exception:
        # try alternative import locations
        from .module import trim_static_frames  # type: ignore
except Exception:
    trim_static_frames = None

_repo_dir = Path(__file__).parent
DEBUG_LOG = _repo_dir / "api_server_debug.log"

def debug_log(msg: str):
    ts = datetime.datetime.now().isoformat()
    pid = os.getpid()
    tid = threading.get_ident()
    line = f"{ts} PID:{pid} TID:{tid} - {msg}"
    try:
        print(line); sys.stdout.flush()
    except Exception:
        pass
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def csv_to_pkl(csv_path: Path, out_pkl: Path, normalize_method: str = 'none', img_shape=(1080, 1920)):
    # Defensive: accept either str or Path and coerce to Path so `.stem` is safe
    try:
        csv_path = Path(csv_path)
    except Exception:
        csv_path = Path(str(csv_path))
    try:
        out_pkl = Path(out_pkl)
    except Exception:
        out_pkl = Path(str(out_pkl))

    debug_log(f"csv_to_pkl: converting {csv_path} to {out_pkl} (normalize={normalize_method}, img_shape={img_shape})")
    
    # 간단한 CSV->PKL 변환: 기존의 csv_to_pkl 로직을 복사해 넣으세요
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    
    debug_log(f"CSV loaded: shape={df.shape}, columns={list(df.columns)[:5]}...")
    
    COCO_NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"]
    missing = []
    for n in COCO_NAMES:
        for s in ["_x","_y","_c"]:
            if f"{n}{s}" not in df.columns:
                missing.append(f"{n}{s}")
    if missing:
        raise ValueError(f"missing cols: {missing[:10]}")
    
    debug_log(f"All required COCO keypoint columns found")
    
    arr = np.stack([
        np.stack([
            df[[f"{name}_x" for name in COCO_NAMES]].values,
            df[[f"{name}_y" for name in COCO_NAMES]].values,
            df[[f"{name}_c" for name in COCO_NAMES]].values
        ], axis=2)
    ], axis=0)[0]
    
    debug_log(f"Keypoint array constructed: shape={arr.shape}")
    
    # CRITICAL FIX: Keep keypoint as (1, T, V, 2) for batch stacking compatibility
    # MMAction2 expects 4D keypoint: (num_person, T, V, 2) where num_person=1
    # Shape: (T, V, 2) -> expand to (1, T, V, 2)
    keypoint = np.expand_dims(arr[:, :, :2], axis=0)  # (1, T, V, 2)
    
    debug_log(f"Keypoint expanded: shape={keypoint.shape}, dtype={keypoint.dtype}")
    
    # Optional normalization: convert pixel coords to 0..1 range using img_shape
    try:
        # Before normalization, attempt to trim static leading/trailing frames
        if trim_static_frames is not None:
            try:
                # trim_static_frames expects (T, V, 2) and returns (start_idx, end_idx)
                # our keypoint currently has shape (1, T, V, 2) -> squeeze to (T, V, 2)
                kp_2d = keypoint[0]  # Get first (and only) person
                start_idx, end_idx = trim_static_frames(kp_2d, fps=30)
                if start_idx > 0 or end_idx < (kp_2d.shape[0] - 1):
                    debug_log(f"Trimming static frames: start={start_idx}, end={end_idx}")
                    keypoint = keypoint[:, start_idx:(end_idx + 1), :, :]  # Trim all persons, all frames
                    keypoint_score = keypoint_score[:, start_idx:(end_idx + 1), :, :]  # Trim score too
            except Exception as e:
                debug_log(f"trim_static_frames failed (non-critical): {e}")
                pass

        if normalize_method == '0to1':
            h, w = img_shape
            kp = keypoint.copy()
            kp[..., 0] = kp[..., 0] / float(w)
            kp[..., 1] = kp[..., 1] / float(h)
            keypoint = kp
            debug_log(f"Normalized keypoints to 0-1 range")
    except Exception as e:
        # If normalization fails, fall back to unnormalized coords
        debug_log(f"Normalization failed (using unnormalized coords): {e}")
        pass

    # CRITICAL: Create keypoint_score with shape (1, T, V) - NO trailing dimension!
    # GenSkeFeat will add [..., None] so we must NOT include it here (would become 5D)
    # Shape: (num_person, T, V) where values are confidence scores
    keypoint_score = np.ones((keypoint.shape[0], keypoint.shape[1], keypoint.shape[2]), dtype=np.float32)
    debug_log(f"Keypoint score created: shape={keypoint_score.shape}, dtype={keypoint_score.dtype}")
    
    # Ensure keypoint is float32 (match make_pkl.py)
    keypoint = keypoint.astype(np.float32)
    
    total_frames = keypoint.shape[1]  # T dimension (after potential trim)
    
    ann = {
        "frame_dir": csv_path.stem,
        "total_frames": int(total_frames),
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,  # CRITICAL: Include keypoint_score (matches make_pkl.py)
        "label": 0,
        "img_shape": tuple(img_shape),
        "original_shape": tuple(img_shape),
        "metainfo": {"frame_dir": csv_path.stem},
    }
    
    debug_log(f"Annotation dict created: frame_dir={ann['frame_dir']}, total_frames={ann['total_frames']}, keypoint.shape={ann['keypoint'].shape}")
    
    data = {"annotations": [ann], "split": {"xsub_val": [csv_path.stem]}}
    
    debug_log(f"PKL data structure: annotations count={len(data['annotations'])}, splits={list(data['split'].keys())}")
    
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f, protocol=4)
    
    debug_log(f"PKL saved successfully: {out_pkl} (size={out_pkl.stat().st_size} bytes)")