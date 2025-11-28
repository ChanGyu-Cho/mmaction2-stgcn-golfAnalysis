import numpy as np
from mmaction.registry import TRANSFORMS

__all__ = ['PreNormalize2D', 'LogGCNInputStats']


@TRANSFORMS.register_module(force=True)
class PreNormalize2D:
    """Normalize 2D skeleton keypoints by centering around the mean.
    
    This transform centers the keypoints by subtracting the mean position,
    which helps with translation invariance during training.
    """
    def __init__(self):
        print("[PreNormalize2D] __init__ called")
        pass
    
    def __call__(self, results):
        print("[PreNormalize2D] __call__ invoked!")
        """Apply pre-normalization to keypoint data.
        
        Expected keypoint shape: (M, T, V, 2) or (M, T, V, 3) where
        M = num_person, T = num_frames, V = num_keypoints, last dim = (x, y) or (x, y, score)
        """
        try:
            if 'keypoint' not in results:
                return results
            
            kp = results['keypoint']
            
            # Handle both (M,T,V,2) and (M,T,V,3) shapes
            if kp.ndim == 4 and kp.shape[-1] >= 2:
                # Normalize only x,y coordinates (not confidence scores)
                coords = kp[..., :2]  # (M, T, V, 2)
                try:
                    pre_mean = float(coords.reshape(-1, 2).mean())
                    pre_std = float(coords.reshape(-1, 2).std())
                    print(f"[PreNormalize2D] before mean={pre_mean:.6f} std={pre_std:.6f}")
                except Exception:
                    pass
                
                # Center by subtracting mean across all valid keypoints
                # Compute mean over M, T, V dimensions
                mean_coords = coords.reshape(-1, 2).mean(axis=0)  # (2,)
                
                # Subtract mean from all coordinates
                kp_normalized = kp.copy()
                kp_normalized[..., :2] = coords - mean_coords
                
                results['keypoint'] = kp_normalized
                try:
                    post_mean = float(kp_normalized[..., :2].reshape(-1, 2).mean())
                    post_std = float(kp_normalized[..., :2].reshape(-1, 2).std())
                    print(f"[PreNormalize2D] after  mean={post_mean:.6f} std={post_std:.6f}")
                except Exception:
                    pass
                
        except Exception as e:
            # If normalization fails, log but don't crash
            print(f"[PreNormalize2D] Warning: normalization failed: {e}")
            pass
        
        return results


@TRANSFORMS.register_module(force=True)
class LogGCNInputStats:
    def __init__(self, prefix: str = "[LogGCNInputStats]"):
        self.prefix = prefix
        print(f"{self.prefix} __init__ called - transform instance created!")

    def __call__(self, results):
        print(f"{self.prefix} __call__ invoked - transform executing!")
        try:
            # FormatGCNInput outputs 'keypoint', not 'inputs'
            # 'inputs' is created by PackActionInputs (which comes after this)
            keypoint = results.get('keypoint', None)
            if keypoint is None:
                print(f"{self.prefix} keypoint missing")
                return results

            if hasattr(keypoint, 'detach'):
                arr = keypoint.detach().cpu().numpy()
            else:
                arr = np.asarray(keypoint)

            # Expected shape after FormatGCNInput: (num_clips, M, T//num_clips, V, C)
            # where M=num_person, T=frames, V=keypoints, C=channels (x,y,score if present)
            stats = {
                'shape': tuple(arr.shape),
                'dtype': str(arr.dtype),
                'min': float(arr.min()) if arr.size else 0.0,
                'max': float(arr.max()) if arr.size else 0.0,
                'mean': float(arr.mean()) if arr.size else 0.0,
                'std': float(arr.std()) if arr.size else 0.0,
            }
            print(f"{self.prefix} keypoint shape={stats['shape']} dtype={stats['dtype']} min={stats['min']:.6f} max={stats['max']:.6f} mean={stats['mean']:.6f} std={stats['std']:.6f}")

            # Check if normalization was applied (mean should be ~0 if PreNormalize2D worked)
            if arr.size > 0:
                xy_coords = arr[..., :2]  # Extract x,y coordinates
                xy_mean = float(xy_coords.mean())
                xy_std = float(xy_coords.std())
                print(f"{self.prefix} xy_coords mean={xy_mean:.6f} std={xy_std:.6f} (expect mean~0 if PreNormalize2D applied)")
            
        except Exception as e:
            print(f"{self.prefix} error: {e}")
        return results
