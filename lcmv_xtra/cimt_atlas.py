"""
CIMT Unified Atlas Extraction (448 ROIs)
========================================
Combines:
  1. Glasser + Tian (414 ROIs) - via gt_extraction logic
  2. Nettekoven (32 ROIs)      - Cerebellum
  3. Custom STN (2 ROIs)       - Subthalamic Nucleus (Left/Right)

Total: 448 ROIs (Indices 0-447)

This module relies on helpers from atlas_extraction.py but implements 
the specific merging logic for the CIMT Lab standard.

Usage:
    from lcmv_xtra.cimt_atlas import cimt_extraction
    
    cimt_tc, cimt_labels = cimt_extraction(
        subject_output_dir=metadata['subject_output'],
        global_subjects_dir=metadata['fsaverage_dir'],
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

# Import shared helpers from the existing atlas_extraction module
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

# CIMT Lab STN Coordinates (Exact match to stn_voxel_extraction.txt)
STN_ROIS = {
    "STN-lh": [-11.89, -14.51, -6.40],
    "STN-rh": [12.53, -13.97, -6.57]
}
STN_RADIUS_MM = 5.0

# =============================================================================
# INTERNAL COMPONENT EXTRACTORS
# =============================================================================

def _extract_nettekoven_component(stc, src, nk_atlas_dir, n_times, logger):
    """
    Extract Nettekoven Cerebellum time courses (32 ROIs).
    Logic adapted from cereb_extraction.txt.
    """
    logger.info(">>> Extracting Nettekoven Cerebellum (414-445)...")
    
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    label_offset = 414
    
    atlas_file = Path(nk_atlas_dir) / "atl-NettekovenSym32_space-MNI_dseg.nii"
    if not atlas_file.exists():
        raise FileNotFoundError(f"Nettekoven atlas not found: {atlas_file}")
        
    cereb_img = nib.load(atlas_file)
    cereb_data = cereb_img.get_fdata()
    
    # Map Sources
    src_coords_mni = _get_mni_coordinates(stc, src)
    vox_coords = _coords_to_voxels(src_coords_mni, cereb_img)
    valid_mask = _filter_valid_voxels(vox_coords, cereb_img.shape)
    valid_indices = np.where(valid_mask)[0]
    valid_voxels = vox_coords[valid_mask]
    
    logger.info(f"Using {len(valid_indices)} sources within cerebellum")
    
    # Assign labels with offset
    labels = np.zeros(len(src_coords_mni), dtype=int)
    shape = cereb_data.shape
    for i, (x, y, z) in enumerate(valid_voxels):
        raw_label = int(cereb_data[x, y, z])
        if raw_label > 0:
            labels[valid_indices[i]] = raw_label + label_offset
            
    # Extract
    n_rois = 32
    time_courses = np.full((n_rois, n_times), np.nan, dtype=np.float32)
    for roi in range(1, n_rois + 1):
        mask = (labels == roi + label_offset)
        if np.any(mask):
            time_courses[roi - 1, :] = np.mean(stc_data[mask, :], axis=0)
            
    return np.nan_to_num(time_courses, nan=0.0)

def _extract_stn_component(stc, src, n_times, logger):
    """
    Extract STN time courses (2 ROIs) using coordinate averaging.
    Logic EXACTLY from stn_voxel_extraction.txt (extract_stn_from_grid).
    """
    logger.info(">>> Extracting Custom STN (446-447)...")
    
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    
    # Use active vertices from STC and coordinates from Source Space
    active_vertices = stc.vertices[0]
    active_coords_mm = src[0]['rr'][active_vertices] * 1000.0
    
    time_courses = []
    
    for roi_name, target_mni in STN_ROIS.items():
        target = np.array(target_mni)
        distances = np.linalg.norm(active_coords_mm - target, axis=1)
        
        selected_indices = np.where(distances <= STN_RADIUS_MM)[0]
        
        if len(selected_indices) == 0:
            logger.warning(f"No voxels within {STN_RADIUS_MM}mm for {roi_name}. Using closest.")
            selected_indices = [np.argmin(distances)]
            
        # Average time series across selected vertices
        roi_data = stc_data[selected_indices, :].mean(axis=0).astype(np.float32)
        time_courses.append(roi_data)
        logger.info(f"  ✅ {roi_name}: Averaged {len(selected_indices)} voxels")
        
    return np.vstack(time_courses)

# =============================================================================
# MAIN UNIFIED FUNCTION
# =============================================================================

def cimt_extraction(subject_output_dir, global_subjects_dir, 
                    stn_radius_mm=5.0, verbose=False):
    """
    Run the full CIMT Atlas extraction (448 ROIs).
    
    This function orchestrates the extraction of Glasser+Tian, Nettekoven, 
    and Custom STN time courses, stacks them, and saves the unified result.
    
    Parameters
    ----------
    subject_output_dir : str or Path
        Path to subject-specific output directory (contains source_estimate_LCMV.h5).
    global_subjects_dir : str or Path
        Path to global resources (fsaverage). MUST contain 'fsaverage-vol-5mm-src.fif'.
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
    subject_output_dir = Path(subject_output_dir)
    global_subjects_dir = Path(global_subjects_dir)
    
    # Setup Logger
    subject_name = subject_output_dir.name
    log = _setup_logger(f'{subject_name}_cimt', subject_output_dir, verbose)
    
    log.info("="*60)
    log.info(f"CIMT Unified Atlas Extraction (448 ROIs): {subject_name}")
    log.info("="*60)
    
    # 1. Validate Inputs
    stc_file = subject_output_dir / "source_estimate_LCMV.h5"
    src_file = global_subjects_dir / "fsaverage-vol-5mm-src.fif"
    
    if not stc_file.exists():
        raise FileNotFoundError(f"Source estimate not found: {stc_file}")
    if not src_file.exists():
        raise FileNotFoundError(
            f"Source space not found: {src_file}\n"
            "Run 'lcmv_xtra.download_fsaverage(path)' to generate this file."
        )
    
    # 2. Load Data Once (Shared across all extractions)
    log.info("Loading source estimate and source space...")
    stc = mne.read_source_estimate(stc_file)
    src = mne.read_source_spaces(src_file)
    
    n_times = stc.data.shape[1]
    log.info(f"Data loaded: {stc.data.shape[0]} sources × {n_times} timepoints")
    
    # 3. Prepare Atlas Paths (From Package Data)
    package_dir = Path(lcmv_xtra.__file__).parent
    data_dir = package_dir / 'data'
    
    gt_atlas_dir = data_dir / 'gt_atlas'
    nk_atlas_dir = data_dir / 'nk_atlas'
    cimt_labels_path = data_dir / 'cimt_atlas' / 'cimt_atlas_labels.csv'
    
    # Verify Files Exist
    if not (gt_atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz").exists():
        raise FileNotFoundError("Bundled GT atlas missing.")
    if not (nk_atlas_dir / "atl-NettekovenSym32_space-MNI_dseg.nii").exists():
        raise FileNotFoundError("Bundled Nettekoven atlas missing.")
    if not cimt_labels_path.exists():
        raise FileNotFoundError("Bundled CIMT labels missing.")
    
    # 4. Execute Extractions
    
    # A. Glasser + Tian (Re-use existing robust logic)
    log.info(">>> Extracting Glasser+Tian (0-413)...")
    # We call the internal helper directly to avoid double-saving intermediate files
    # But to keep it simple and consistent, we can just re-implement the core logic 
    # or call extract_gt_ts and ignore its save step if we refactor it. 
    # For now, let's inline the core extraction logic to ensure we get the array directly.
    
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    
    # GT Logic Inline (Simplified for stacking)
    glasser_file = gt_atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz"
    tian_file = gt_atlas_dir / "tian_subcortex_54_MNI152NLin6Asym.nii"
    glasser_img = nib.load(glasser_file)
    tian_img = nib.load(tian_file)
    tian_resampled = image.resample_to_img(tian_img, glasser_img, interpolation='nearest', force_resample=True)
    g_data = glasser_img.get_fdata()
    t_data = tian_resampled.get_fdata()
    t_data = np.where(t_data > 0, t_data + 360, 0)
    gt_data = g_data + t_data
    gt_img = nib.Nifti1Image(gt_data, affine=glasser_img.affine)
    
    src_coords_mni = _get_mni_coordinates(stc, src)
    vox_coords = _coords_to_voxels(src_coords_mni, gt_img)
    labels = np.zeros(len(src_coords_mni), dtype=int)
    shape = gt_data.shape
    gt_array = gt_img.get_fdata()
    for i, (x, y, z) in enumerate(vox_coords):
        if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
            labels[i] = int(gt_array[x, y, z])
            
    n_rois_gt = 414
    gt_tc = np.full((n_rois_gt, n_times), np.nan, dtype=np.float32)
    for roi in range(1, n_rois_gt + 1):
        mask = (labels == roi)
        if np.any(mask):
            gt_tc[roi - 1, :] = np.mean(stc_data[mask, :], axis=0)
    gt_tc = np.nan_to_num(gt_tc, nan=0.0)
    
    # B. Nettekoven
    cereb_tc = _extract_nettekoven_component(stc, src, nk_atlas_dir, n_times, log)
    
    # C. STN
    stn_tc = _extract_stn_component(stc, src, n_times, log)
    
    # 5. Merge & Validate
    log.info(">>> Stacking components into 448-ROI array...")
    
    # Check Time Alignment
    if not (gt_tc.shape[1] == cereb_tc.shape[1] == stn_tc.shape[1]):
        raise ValueError("Time dimension mismatch during stacking!")
        
    # Stack: GT (0-413) -> Cereb (414-445) -> STN (446-447)
    cimt_tc = np.vstack([gt_tc, cereb_tc, stn_tc])
    
    # Load Pre-defined Labels
    log.info(f"Loading unified labels from: {cimt_labels_path.name}")
    cimt_labels = pd.read_csv(cimt_labels_path)
    
    # Final Validation
    assert cimt_tc.shape[0] == 448, f"Row count mismatch: {cimt_tc.shape[0]}"
    assert len(cimt_labels) == 448, "Label count mismatch!"
    assert list(cimt_labels['index']) == list(range(448)), "Indices not sequential!"
    
    # 6. Save Outputs
    out_file = subject_output_dir / "cimt_time_courses.npy"
    np.save(out_file, cimt_tc)
    cimt_labels.to_csv(subject_output_dir / "cimt_atlas_labels.csv", index=False)
    
    log.info("✅ CIMT Extraction Complete!")
    log.info(f"   Shape: {cimt_tc.shape}")
    log.info(f"   Saved: {out_file.name}")
    log.info("="*60 + "\n")
    
    return cimt_tc, cimt_labels
