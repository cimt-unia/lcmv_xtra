"""
CIMT Unified Atlas Extraction (448 ROIs)
========================================
Combines:
  1. Glasser + Tian (414 ROIs)
  2. Nettekoven (32 ROIs)
  3. Custom STN (2 ROIs)

All atlas files are bundled within the lcmv_xtra package.
Only the fsaverage source space (fsaverage_dir) is user-specific.

Usage:
    from lcmv_xtra.cimt_atlas import cimt_extraction
    
    cimt_tc, cimt_labels = cimt_extraction(
        subject_output_dir=metadata['subject_output'],
        fsaverage_dir=metadata['fsaverage_dir'],
        verbose=True
    )
"""

import mne
import logging
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import image
import lcmv_xtra


from lcmv_xtra.atlas_extraction import (
    _setup_logger, 
    _get_mni_coordinates, 
    _coords_to_voxels, 
    _filter_valid_voxels,
    extract_gt_ts
)

# =============================================================================
# CONFIG
# =============================================================================

STN_ROIS = {
    "STN-lh": [-11.89, -14.51, -6.40],
    "STN-rh": [12.53, -13.97, -6.57]
}
STN_RADIUS_MM = 5.0

# =============================================================================
# INTERNAL COMPONENT EXTRACTORS
# =============================================================================

def _extract_nettekoven_ts(stc, src, nk_atlas_dir, subject_output, label_offset=414, logger=None):
    """Extract Nettekoven time courses (32 ROIs)."""
    log = logger or logging.getLogger(__name__)
    log.info("Starting Nettekoven cerebellar atlas extraction")
    
    # Handle complex-valued source estimates consistently
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    n_times = stc_data.shape[1]
    
    atlas_file = Path(nk_atlas_dir) / "atl-NettekovenSym32_space-MNI_dseg.nii"
    csv_file = Path(nk_atlas_dir) / "nettekoven_roi_labels.csv"
    
    if not atlas_file.exists():
        raise FileNotFoundError(f"Cerebellar atlas not found: {atlas_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"ROI labels file not found: {csv_file}")
    
    cereb_img = nib.load(atlas_file)
    cereb_data = cereb_img.get_fdata()
    
    src_coords_mni = _get_mni_coordinates(stc, src)
    vox_coords = _coords_to_voxels(src_coords_mni, cereb_img)
    valid_mask = _filter_valid_voxels(vox_coords, cereb_img.shape)
    valid_indices = np.where(valid_mask)[0]
    valid_voxels = vox_coords[valid_mask]
    
    log.info(f"Using {len(valid_indices)} sources within cerebellum")
    
    labels = np.zeros(len(src_coords_mni), dtype=int)
    shape = cereb_data.shape
    for i, (x, y, z) in enumerate(valid_voxels):
        raw_label = int(cereb_data[x, y, z])
        if raw_label > 0:
            labels[valid_indices[i]] = raw_label + label_offset
            
    n_rois = 32
    time_courses = np.full((n_rois, n_times), np.nan, dtype=np.float32)
    
    for roi in range(1, n_rois + 1):
        mask = (labels == roi + label_offset)
        if np.any(mask):
            time_courses[roi - 1, :] = np.mean(stc_data[mask, :], axis=0)
    
    roi_df = pd.read_csv(csv_file)
    if len(roi_df) != 32:
        raise ValueError(f"Expected 32 ROIs, got {len(roi_df)}")
    roi_names = roi_df['roi_name'].tolist()
    
    return np.nan_to_num(time_courses, nan=0.0), roi_names


def _extract_stn_ts(stc, src, n_times, logger=None):
    """Extract STN time courses (2 ROIs) using coordinate averaging."""
    log = logger or logging.getLogger(__name__)
    log.info("Starting Custom STN voxel extraction")
    
    # FIX: Reference stc.data (not stc_data) in the else clause
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    
    active_vertices = stc.vertices[0]
    active_coords_mm = src[0]['rr'][active_vertices] * 1000.0
    
    time_courses = []
    roi_names = []
    
    for roi_name, target_mni in STN_ROIS.items():
        target = np.array(target_mni)
        distances = np.linalg.norm(active_coords_mm - target, axis=1)
        
        selected_indices = np.where(distances <= STN_RADIUS_MM)[0]
        
        if len(selected_indices) == 0:
            log.warning(f"No voxels within {STN_RADIUS_MM}mm for {roi_name}. Using closest.")
            selected_indices = [np.argmin(distances)]
            
        roi_data = stc_data[selected_indices, :].mean(axis=0).astype(np.float32)
        time_courses.append(roi_data)
        roi_names.append(roi_name)
        log.info(f"  ✅ {roi_name}: Averaged {len(selected_indices)} voxels")
        
    return np.vstack(time_courses), roi_names



# =============================================================================
# MAIN UNIFIED FUNCTION
# =============================================================================

def cimt_extraction(subject_output_dir, fsaverage_dir, 
                    stn_radius_mm=5.0, verbose=False):
    """
    Run the full CIMT Atlas extraction (448 ROIs).
    
    Parameters
    ----------
    subject_output_dir : str or Path
        Path to subject output (contains source_estimate_LCMV.h5).
    fsaverage_dir : str or Path
        Path to fsaverage resources. MUST contain 'fsaverage-vol-5mm-src.fif'.
    stn_radius_mm : float
        Radius for STN voxel averaging (default 5.0).
    verbose : bool
        Enable console logging.
        
    Returns
    -------
    cimt_tc : np.ndarray
        Array of shape (448, n_times).
    cimt_labels : pd.DataFrame
        DataFrame loaded from package data (indices 0-447).
    """
    global STN_RADIUS_MM
    if stn_radius_mm != STN_RADIUS_MM:
        STN_RADIUS_MM = stn_radius_mm
    
    subject_output_dir = Path(subject_output_dir)
    fsaverage_dir = Path(fsaverage_dir)
    
    subject_name = subject_output_dir.name
    log = _setup_logger(f'{subject_name}_cimt', subject_output_dir, verbose)
    
    log.info("="*60)
    log.info(f"CIMT Unified Atlas Extraction (448 ROIs): {subject_name}")
    log.info("="*60)
    
    # 1. Validate Inputs
    stc_file = subject_output_dir / "source_estimate_LCMV.h5"
    src_file = fsaverage_dir / "fsaverage-vol-5mm-src.fif"
    
    if not stc_file.exists():
        raise FileNotFoundError(f"Source estimate not found: {stc_file}")
    if not src_file.exists():
        raise FileNotFoundError(
            f"Source space not found: {src_file}\n"
            f"Ensure fsaverage_dir points to the folder containing fsaverage-vol-5mm-src.fif"
        )
    
    # 2. Load Data
    log.info("Loading source estimate and source space...")
    stc = mne.read_source_estimate(stc_file)
    src = mne.read_source_spaces(src_file)
    n_times = stc.data.shape[1]
    log.info(f"Data loaded: {stc.data.shape[0]} sources × {n_times} timepoints")
    
    # 3. DYNAMICALLY LOCATE BUNDLED ATLAS FILES
    package_dir = Path(lcmv_xtra.__file__).parent
    data_dir = package_dir / 'data'
    
    gt_atlas_dir = data_dir / 'gt_atlas'
    nk_atlas_dir = data_dir / 'nk_atlas'
    cimt_labels_path = data_dir / 'cimt_atlas' / 'cimt_atlas_labels.csv'
    
    if not (gt_atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz").exists():
        raise FileNotFoundError("Bundled GT atlas missing.")
    if not (nk_atlas_dir / "atl-NettekovenSym32_space-MNI_dseg.nii").exists():
        raise FileNotFoundError("Bundled Nettekoven atlas missing.")
    if not cimt_labels_path.exists():
        raise FileNotFoundError("Bundled CIMT labels missing.")
    
    # 4. Execute Extractions
    
    # A. Glasser + Tian
    log.info(">>> Extracting Glasser+Tian (0-413)...")
    gt_tc, gt_names = extract_gt_ts(
        stc=stc, src=src, gt_atlas_dir=gt_atlas_dir, 
        subject_output=subject_output_dir, logger=log
    )
    
    # B. Nettekoven
    log.info(">>> Extracting Nettekoven Cerebellum (414-445)...")
    cereb_tc, cereb_names = _extract_nettekoven_ts(
        stc=stc, src=src, nk_atlas_dir=nk_atlas_dir, 
        subject_output=subject_output_dir, label_offset=414, logger=log
    )
    
    # C. STN
    log.info(">>> Extracting Custom STN (446-447)...")
    stn_tc, stn_names = _extract_stn_ts(
        stc=stc, src=src, n_times=n_times, logger=log
    )
    
    # 5. Merge & Validate
    log.info(">>> Stacking components into 448-ROI array...")
    if not (gt_tc.shape[1] == cereb_tc.shape[1] == stn_tc.shape[1]):
        raise ValueError("Time dimension mismatch!")
        
    cimt_tc = np.vstack([gt_tc, cereb_tc, stn_tc])
    
    # Load Pre-defined Labels
    log.info(f"Loading unified labels from: {cimt_labels_path.name}")
    cimt_labels = pd.read_csv(cimt_labels_path)
    
    # Final Validation
    assert cimt_tc.shape[0] == 448, f"Row count mismatch: {cimt_tc.shape[0]}"
    assert len(cimt_labels) == 448, "Label count mismatch!"
    assert list(cimt_labels['index']) == list(range(448)), "Indices not sequential!"
    
    # Zero-variance ROI quality check
    zero_var_mask = np.std(cimt_tc, axis=1) < 1e-12
    n_zero_var = int(zero_var_mask.sum())
    if n_zero_var > 0:
        zero_var_names = cimt_labels.loc[zero_var_mask, 'roi_name'].tolist()
        log.warning(
            f"⚠️  {n_zero_var}/448 ROIs have zero variance "
            f"(no valid sources assigned or silent region):"
        )
        for name in zero_var_names[:10]:
            log.warning(f"   - {name}")
        if n_zero_var > 10:
            log.warning(f"   ... and {n_zero_var - 10} more")
    else:
        log.info("✓ All 448 ROIs have non-zero variance")
    
    # 6. Save Outputs
    out_file = subject_output_dir / "cimt_time_courses.npy"
    np.save(out_file, cimt_tc)
    cimt_labels.to_csv(subject_output_dir / "cimt_atlas_labels.csv", index=False)
    
    log.info("✅ CIMT Extraction Complete!")
    log.info(f"   Shape: {cimt_tc.shape}")
    log.info("="*60 + "\n")

    return cimt_tc, cimt_labels


