#!/usr/bin/env python3
"""
Debug script to analyze CSV keypoint data quality.
Run this on both local (good) and server (bad) CSV files to compare.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_csv(csv_path):
    """Analyze a skeleton CSV file for potential issues."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(csv_path).name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)} ({df.columns[0]}, {df.columns[1]}, ...)")
    
    # Extract x, y coordinates
    COCO_NAMES = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder',
                  'LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee',
                  'RKnee','LAnkle','RAnkle']
    
    x_cols = [f"{n}_x" for n in COCO_NAMES]
    y_cols = [f"{n}_y" for n in COCO_NAMES]
    c_cols = [f"{n}_c" for n in COCO_NAMES]
    
    x_data = df[x_cols].values
    y_data = df[y_cols].values
    c_data = df[c_cols].values
    
    # Check for zeros/missing
    zero_x_ratio = (x_data == 0).sum() / x_data.size
    zero_y_ratio = (y_data == 0).sum() / y_data.size
    zero_c_ratio = (c_data == 0).sum() / c_data.size
    
    print(f"\n📊 Zero Value Analysis:")
    print(f"  X coordinates with 0: {zero_x_ratio*100:.2f}%")
    print(f"  Y coordinates with 0: {zero_y_ratio*100:.2f}%")
    print(f"  Confidence with 0: {zero_c_ratio*100:.2f}%")
    
    # Check coordinate ranges
    x_min, x_max = x_data[x_data > 0].min() if (x_data > 0).any() else 0, x_data.max()
    y_min, y_max = y_data[y_data > 0].min() if (y_data > 0).any() else 0, y_data.max()
    c_min, c_max = c_data[c_data > 0].min() if (c_data > 0).any() else 0, c_data.max()
    
    print(f"\n📐 Coordinate Ranges:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Confidence: [{c_min:.4f}, {c_max:.4f}]")
    
    # Check for sudden jumps (no interpolation)
    x_diff = np.abs(np.diff(x_data, axis=0))
    y_diff = np.abs(np.diff(y_data, axis=0))
    
    x_diff_mean = x_diff.mean()
    y_diff_mean = y_diff.mean()
    x_diff_max = x_diff.max()
    y_diff_max = y_diff.max()
    
    print(f"\n🔀 Frame-to-Frame Movement:")
    print(f"  X mean diff: {x_diff_mean:.2f} px, max diff: {x_diff_max:.2f} px")
    print(f"  Y mean diff: {y_diff_mean:.2f} px, max diff: {y_diff_max:.2f} px")
    
    # Check for static frames (potential trimming issue)
    movement_threshold = 5.0  # pixels
    static_frames = ((x_diff < movement_threshold).all(axis=1) & 
                    (y_diff < movement_threshold).all(axis=1))
    static_ratio = static_frames.sum() / len(static_frames)
    
    print(f"\n⏸️  Static Frames (movement < {movement_threshold}px):")
    print(f"  Count: {static_frames.sum()} / {len(static_frames)} ({static_ratio*100:.2f}%)")
    
    # Check first/last 5 frames (common static regions)
    print(f"\n📍 First 5 frames movement:")
    for i in range(min(5, len(x_diff))):
        move = np.sqrt(x_diff[i]**2 + y_diff[i]**2).mean()
        print(f"  Frame {i}→{i+1}: {move:.2f} px avg movement")
    
    print(f"\n📍 Last 5 frames movement:")
    for i in range(max(0, len(x_diff)-5), len(x_diff)):
        move = np.sqrt(x_diff[i]**2 + y_diff[i]**2).mean()
        print(f"  Frame {i}→{i+1}: {move:.2f} px avg movement")
    
    # Sample data preview
    print(f"\n🔍 Sample Data (first frame, first 5 keypoints):")
    for i, name in enumerate(COCO_NAMES[:5]):
        print(f"  {name:12s}: x={x_data[0, i]:7.2f}, y={y_data[0, i]:7.2f}, c={c_data[0, i]:.4f}")
    
    # Check for normalized data (values in [0,1])
    if x_max <= 1.0 and y_max <= 1.0:
        print(f"\n⚠️  WARNING: Data appears to be normalized to [0,1] range!")
        print(f"   Expected pixel coordinates (e.g., 0-1920), found max x={x_max:.4f}, y={y_max:.4f}")
    
    # Check for suspicious patterns
    if zero_x_ratio > 0.3 or zero_y_ratio > 0.3:
        print(f"\n⚠️  WARNING: High percentage of zero coordinates!")
        print(f"   This suggests missing keypoints or poor detection.")
    
    if static_ratio > 0.5:
        print(f"\n⚠️  WARNING: More than 50% frames are static!")
        print(f"   trim_static_frames should remove these.")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_csv_analysis.py <csv_file>")
        print("Example: python debug_csv_analysis.py /tmp/skeleton.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    analyze_csv(csv_path)
