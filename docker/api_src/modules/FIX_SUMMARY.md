# STGCN Test API - Error Fix Summary

## Problem Analysis
The error occurred in `stgcn_server.py` when calling `runner.test()`:

```
File ".../mmengine/dataset/base_dataset.py", line 403, in __getitem__
    data = self.prepare_data(idx)
```

This indicates that the dataset was unable to properly load and prepare the annotation data. The issue stemmed from three main areas:

1. **ann_file Configuration**: The annotation file path wasn't being reliably set before `Runner.from_cfg()` was called
2. **PKL Validation**: The generated PKL file wasn't being validated for correctness
3. **Error Handling**: Poor error messages made it difficult to diagnose what was failing

## Fixes Implemented

### 1. **Enhanced ann_file Configuration** (`stgcn_tester.py`)

**Before:**
```python
try:
    if hasattr(cfg, 'test_dataloader'):
        try:
            cfg.test_dataloader.dataset.ann_file = str(ann_pkl_path)
        except Exception:
            try:
                cfg.test_dataloader['dataset']['ann_file'] = str(ann_pkl_path)
            except Exception:
                pass  # Silent failure!
except Exception:
    pass
```

**After:**
- Added explicit error logging when ann_file setting fails
- Raises ValueError if both attribute-style and dict-style access fail
- Provides clear debug messages about what was attempted

**Impact:** Now you'll know exactly when and why ann_file configuration fails.

### 2. **PKL File Validation** (`stgcn_tester.py`)

Added comprehensive validation after PKL creation:
- Checks that PKL file exists
- Verifies PKL contains a dict with 'annotations' key
- Validates annotations is a non-empty list
- Confirms first annotation has required 'keypoint' field
- Logs keypoint shape and dtype for debugging

**Impact:** Catches PKL generation errors before they cause dataset loading failures.

### 3. **Enhanced csv_to_pkl Logging** (`utils.py`)

Added detailed logging throughout the CSV→PKL conversion process:
- CSV loading (shape, columns)
- COCO keypoint validation
- Keypoint array construction
- Shape transformations (1, T, V, 2) format
- Normalization steps
- Final PKL save confirmation

**Impact:** Easy to trace exactly where in the conversion process issues occur.

### 4. **Better Runner Error Handling** (`stgcn_tester.py`)

Added explicit try-catch blocks:
- `Runner.from_cfg()` now has detailed error reporting
- `runner.test()` now captures and logs full traceback
- Additional context about dataloader/dataset types
- More informative error messages instead of silent failures

**Impact:** When errors occur, you get the full picture of what went wrong.

## Key Configuration Points

### CSV Requirements
The CSV must contain COCO 17-keypoint columns in this format:
```
Nose_x, Nose_y, Nose_c,
LEye_x, LEye_y, LEye_c,
REye_x, REye_y, REye_c,
... (17 keypoints total)
```

### PKL Structure
The generated PKL must have this structure:
```python
{
    "annotations": [
        {
            "frame_dir": "filename",
            "total_frames": N,
            "keypoint": ndarray(1, T, V, 2),  # (num_person=1, time, vertices, xy)
            "label": 0,
            "img_shape": (1080, 1920),
            "original_shape": (1080, 1920),
            "metainfo": {"frame_dir": "filename"}
        }
    ],
    "split": {
        "xsub_val": ["filename"]
    }
}
```

### Keypoint Shape
- Input CSV: (frame_count, 17 keypoints × 3 channels)
- After processing: (1, T, V, 2) where:
  - 1 = num_person
  - T = number of frames
  - V = 17 vertices (COCO)
  - 2 = (x, y) coordinates

## Testing the Fix

To verify the fix works, check the debug log for:

1. ✅ "Converting CSV to PKL: ..." - CSV loading started
2. ✅ "CSV loaded: shape=..." - CSV successfully read
3. ✅ "All required COCO keypoint columns found" - Columns validated
4. ✅ "Keypoint expanded: shape=..." - Array reshaping succeeded
5. ✅ "PKL saved successfully: ... (size=...)" - File created
6. ✅ "PKL validation successful: ..." - Content verified
7. ✅ "Setting ann_file to: ..." - Config update attempted
8. ✅ "Successfully set cfg.test_dataloader.dataset.ann_file" - Config succeeded
9. ✅ "Creating Runner from config..." - Runner initialization started
10. ✅ "Runner created successfully" - Ready for inference
11. ✅ "Starting runner.test()..." - Inference running

If any step fails, the error will be clearly logged with context.

## Debugging Guide

### If PKL validation fails:
- Check CSV structure and column names
- Verify CSV has data in all required COCO columns
- Check for NaN or invalid values

### If ann_file setting fails:
- Verify config structure matches MMAction2 expectations
- Check that test_dataloader.dataset exists in config

### If runner.test() fails:
- Check that keypoint shape is (1, T, V, 2)
- Verify label and img_shape are present in annotation
- Look for data type mismatches in MMAction2 transforms

## Files Modified
1. `/mmaction2/docker/api_src/modules/stgcn_tester.py` - Configuration and error handling
2. `/mmaction2/docker/api_src/modules/utils.py` - PKL generation and validation
