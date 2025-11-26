#!/usr/bin/env python3
"""Test MMACTION API with crop_bbox parameter to verify img_shape update."""

import requests
import base64
import json
from pathlib import Path

# Test CSV from skeleton_metric-client (OpenPose output)
test_csv = Path(r"d:\Jabez\golf\containers\client\skeleton_mertic-client\input\c4282d5c-5091-7071-aa3d-6f5d97e2fd8e\2d\skeleton2d.csv")

if not test_csv.exists():
    print(f"ERROR: Test CSV not found: {test_csv}")
    exit(1)

# Read CSV and encode to base64
with open(test_csv, 'r', encoding='utf-8') as f:
    csv_content = f.read()

csv_b64 = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')

# Test 1: Without crop_bbox (old behavior - should use default 1080x1920)
print("=" * 60)
print("Test 1: API call WITHOUT crop_bbox (default img_shape)")
print("=" * 60)

payload1 = {
    'csv_base64': csv_b64
}

resp1 = requests.post('http://localhost:19031/mmaction_stgcn_test', json=payload1, timeout=120)
print(f"Status Code: {resp1.status_code}")
if resp1.ok:
    result1 = resp1.json()
    print(f"Response (without crop_bbox):")
    print(json.dumps(result1, indent=2))
else:
    print(f"ERROR: {resp1.text}")

print("\n" + "=" * 60)
print("Test 2: API call WITH crop_bbox (should update img_shape)")
print("=" * 60)

# Simulate a crop: x=100, y=50, w=800, h=600
# This should make img_shape = (600, 800) instead of (1080, 1920)
payload2 = {
    'csv_base64': csv_b64,
    'crop_bbox': [100, 50, 800, 600]
}

resp2 = requests.post('http://localhost:19031/mmaction_stgcn_test', json=payload2, timeout=120)
print(f"Status Code: {resp2.status_code}")
if resp2.ok:
    result2 = resp2.json()
    print(f"Response (with crop_bbox):")
    print(json.dumps(result2, indent=2))
    
    # Compare predictions
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("=" * 60)
    
    if 'pred_score' in result1 and 'pred_score' in result2:
        score1 = result1['pred_score']
        score2 = result2['pred_score']
        print(f"Prediction WITHOUT crop: {score1} -> label={result1.get('pred_label')}")
        print(f"Prediction WITH crop:    {score2} -> label={result2.get('pred_label')}")
        
        # Check if predictions changed (should be different if img_shape affects features)
        if score1 != score2:
            print("\n✅ SUCCESS: Predictions changed when crop_bbox provided!")
            print("   This confirms img_shape is being updated correctly.")
        else:
            print("\n⚠️ WARNING: Predictions identical despite crop_bbox!")
            print("   img_shape may not be affecting feature extraction.")
else:
    print(f"ERROR: {resp2.text}")
