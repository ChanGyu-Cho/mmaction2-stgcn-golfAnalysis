import numpy as np
from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class PreNormalize2D:
    """Normalize 2D skeleton keypoints by centering around the mean.
    
    This transform centers the keypoints by subtracting the mean position,
    which helps with translation invariance during training.
    """
    def __init__(self):
        pass
    
    def __call__(self, results):
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
                
                # Center by subtracting mean across all valid keypoints
                # Compute mean over M, T, V dimensions
                mean_coords = coords.reshape(-1, 2).mean(axis=0)  # (2,)
                
                # Subtract mean from all coordinates
                kp_normalized = kp.copy()
                kp_normalized[..., :2] = coords - mean_coords
                
                results['keypoint'] = kp_normalized
                
        except Exception:
            # If normalization fails, silently skip (avoid logging spam during training)
            pass
        
        return results


@TRANSFORMS.register_module(force=True)
class LogGCNInputStats:
    def __init__(self, prefix: str = "[LogGCNInputStats]"):
        self.prefix = prefix

    def __call__(self, results):
        try:
            # FormatGCNInput outputs 'keypoint', not 'inputs'
            # 'inputs' is created by PackActionInputs (which comes after this)
            keypoint = results.get('keypoint', None)
            if keypoint is None:
                return results

            if hasattr(keypoint, 'detach'):
                arr = keypoint.detach().cpu().numpy()
            else:
                arr = np.asarray(keypoint)

            # Expected shape after FormatGCNInput: (num_clips, M, T//num_clips, V, C)
            # where M=num_person, T=frames, V=keypoints, C=channels (x,y,score if present)
            # (Logging disabled to avoid spam during training)
            
        except Exception:
            # Silently skip on error (avoid logging spam)
            pass
        return results
