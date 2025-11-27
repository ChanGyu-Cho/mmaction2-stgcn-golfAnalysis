import numpy as np
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LogGCNInputStats:
    def __init__(self, prefix: str = "[LogGCNInputStats]"):
        self.prefix = prefix

    def __call__(self, results):
        try:
            inputs = results.get('inputs', None)
            if inputs is None:
                print(f"{self.prefix} inputs missing")
                return results

            if hasattr(inputs, 'detach'):
                arr = inputs.detach().cpu().numpy()
            else:
                arr = np.asarray(inputs)

            # Expect shape (N, C, T, V, M) for GCN input; handle generically
            stats = {
                'shape': tuple(arr.shape),
                'dtype': str(arr.dtype),
                'min': float(arr.min()) if arr.size else 0.0,
                'max': float(arr.max()) if arr.size else 0.0,
                'mean': float(arr.mean()) if arr.size else 0.0,
                'std': float(arr.std()) if arr.size else 0.0,
            }
            print(f"{self.prefix} shape={stats['shape']} dtype={stats['dtype']} min={stats['min']:.6f} max={stats['max']:.6f} mean={stats['mean']:.6f} std={stats['std']:.6f}")

            # Per-dimension stats if 5D
            if arr.ndim >= 3:
                # Over T dimension if present
                try:
                    t_axis = 2 if arr.ndim >= 3 else None
                    if t_axis is not None and arr.shape[t_axis] > 0:
                        t_mean = arr.mean(axis=t_axis)
                        print(f"{self.prefix} per-T mean shape={tuple(t_mean.shape)} mean_range=({float(t_mean.min()):.6f},{float(t_mean.max()):.6f})")
                except Exception:
                    pass
        except Exception as e:
            print(f"{self.prefix} error: {e}")
        return results
