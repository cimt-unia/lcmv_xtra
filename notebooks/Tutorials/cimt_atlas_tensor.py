# =============================================================================
# CIMT Tensor — How to Build, Load, and Inspect
# =============================================================================
"""
This script shows how to create a CIMT tensor from EEG data and understand
its structure. A "tensor" here is a 3D NumPy array: (subjects, 448 ROIs, timepoints).

Requirements:
  - Cleaned EEG .fif files with valid montages (nasion, LPA, RPA fiducials)
  - fsaverage directory with BEM + 5mm volume source space
  - lcmv_xtra installed
"""

import lcmv_xtra as lx
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
FS_DIR       = Path("path/to/fsaverage")       # fsaverage with BEM + source space
OUTPUT_DIR   = Path("path/to/output")           # where .npz tensors will be saved
PROJECT_BASE = Path("path/to/project_root")     # root of your BIDS-like project

# =============================================================================
# STEP 1: Create a subject list
# =============================================================================
"""
assemble_tensor() needs a DataFrame with two columns:
  - subject_id : a unique name for each subject
  - fif_path   : absolute path to the cleaned .fif file

There are two ways to build this DataFrame:
"""

# --- Method A: Manual list (works for any file naming convention) ---
df = lx.make_subject_list(
    paths=[
        "/data/sub-01_resting_cleaned.fif",
        "/data/sub-02_resting_cleaned.fif",
        "/data/sub-03_resting_cleaned.fif",
    ],
    ids=["sub-01", "sub-02", "sub-03"],   # optional: defaults to sub-00, sub-01, ...
)
print("Subject list:")
print(df)
#   subject_id                              fif_path
# 0     sub-01   /data/sub-01_resting_cleaned.fif
# 1     sub-02   /data/sub-02_resting_cleaned.fif
# 2     sub-03   /data/sub-03_resting_cleaned.fif

# --- Method B: Auto-scan a directory (simpler if files are organized) ---
df = lx.scan_eeg_paths(
    Path("path/to/cleaned/files"),
    pattern="*_resting_cleaned.fif",
)
# Automatically finds all matching .fif files and extracts subject IDs
# from parent folder names (e.g., sub-01/sub-01_resting_cleaned.fif → sub-01).

# =============================================================================
# STEP 2: Build the tensor
# =============================================================================
"""
This runs the full pipeline per subject:
  1. Load .fif file
  2. Coregister electrodes to fsaverage head model
  3. Compute LCMV beamformer source estimates
  4. Extract time courses from all 448 CIMT atlas ROIs
  5. Resample to a common sampling rate (250 Hz default)
  6. Stack all subjects into one 3D array

Output: study_resting.npz
"""

tensor_path = lx.assemble_tensor(
    data_index=df,             # your subject DataFrame
    fs_dir=FS_DIR,             # where fsaverage lives
    output_dir=OUTPUT_DIR,     # where to save the .npz
    task_name="resting",       # → study_resting.npz
    project_base=PROJECT_BASE,
    n_jobs=-1,                 # CPU cores (-1 = all available)
    target_sfreq=250.0,        # resample all subjects to 250 Hz
    verbose=True,              # show progress per subject
)
print(f"Tensor saved to: {tensor_path}")

# =============================================================================
# STEP 3: Load and understand the shape
# =============================================================================
tensor = np.load(OUTPUT_DIR / "study_resting.npz", allow_pickle=True)

# The .npz file contains three arrays:
#   'data'        — the actual brain activity, shape (subjects, 448, timepoints)
#   'subject_ids' — string array of subject names
#   'sfreq'       — sampling rate in Hz

print(f"\nTensor contents:")
print(f"  data shape:    {tensor['data'].shape}")
print(f"  subject_ids:   {tensor['subject_ids']}")
print(f"  sfreq:         {tensor['sfreq']} Hz")

# Example output for 3 subjects with 1500 timepoints at 250 Hz:
#   data shape:    (3, 448, 1500)
#   subject_ids:   ['sub-01' 'sub-02' 'sub-03']
#   sfreq:         250.0 Hz

# =============================================================================
# STEP 4: How to index the tensor
# =============================================================================

# A. One subject, all ROIs, all timepoints
subject_0 = tensor['data'][0]            # shape: (448, 1500)
print(f"\nSubject 0 shape: {subject_0.shape}")

# B. One subject, one ROI, all timepoints (1D time series)
subject_0_left_m1 = tensor['data'][0, 7] # index 7 = L_4_ROI = left M1
print(f"Subject 0, left M1: {subject_0_left_m1.shape}")  # (1500,)

# C. All subjects, one ROI, all timepoints
all_left_m1 = tensor['data'][:, 7, :]    # shape: (3, 1500)
print(f"All subjects, left M1: {all_left_m1.shape}")

# D. All subjects, all ROIs, first 100 timepoints
snapshot = tensor['data'][:, :, :100]    # shape: (3, 448, 100)

# E. Group average across subjects
group_mean = tensor['data'].mean(axis=0) # shape: (448, 1500)
print(f"Group mean shape: {group_mean.shape}")

# =============================================================================
# STEP 5: Find which index corresponds to which brain region
# =============================================================================
labels = pd.read_csv(
    Path(lx.__file__).parent / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'
)
print(f"\nCIMT atlas: {len(labels)} ROIs")
print(f"First 5:\n{labels.head()[['index', 'roi_name', 'hemisphere', 'region_full_name']]}")

# Quick lookups:
print(f"\nLeft M1: {labels[labels['roi_name'] == 'L_4_ROI'][['index', 'roi_name']].values}")
print(f"STN:     {labels[labels['roi_name'].str.contains('STN')][['index', 'roi_name']].values}")
print(f"Putamen: {labels[labels['roi_name'].str.contains('PUT')][['index', 'roi_name']].values}")

# =============================================================================
# QUICK REFERENCE: Tensor indexing
# =============================================================================
"""
tensor['data']                        → (subjects, 448, timepoints)
tensor['data'][subject]               → (448, timepoints)
tensor['data'][subject, roi]          → (timepoints,)
tensor['data'][:, roi, :]             → (subjects, timepoints)
tensor['data'][:, :, t_start:t_end]   → (subjects, 448, window)
tensor['data'].mean(axis=0)           → (448, timepoints)  — group average
tensor['data'].std(axis=0)            → (448, timepoints)  — group std
"""
