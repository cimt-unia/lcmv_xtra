# lcmv_xtra/cimt_atlas.py
"""
CIMT Unified Atlas Extraction (448 ROIs)
========================================
Uses the pre-computed unified CIMT atlas (CIMT_448ROIs_atlas.nii.gz)
containing all 448 ROIs in a single NIfTI file:
  - Glasser + Tian (0-413)
  - Nettekoven Cerebellum (414-445)
  - Custom STN (446-447)

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
from typing import Optional, Tuple

import lcmv_xtra
from lcmv_xtra.atlas_extraction import (
    _setup_logger,
    _get_mni_coordinates,
    _coords_to_voxels,
    _filter_valid_voxels,
)


def cimt_extraction(
    subject_output_dir: Path,
    fsaverage_dir: Path,
    stc_filename: str = 'source_estimate_LCMV.h5',
    stn_radius_mm: float = 5.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Run the full CIMT Atlas extraction (448 ROIs) using the unified atlas.

    Parameters
    ----------
    subject_output_dir : str or Path
        Path to subject output directory.
    fsaverage_dir : str or Path
        Path to fsaverage resources. MUST contain 'fsaverage-vol-5mm-src.fif'.
    stc_filename : str
        Name of the source estimate file within subject_output_dir.
        Defaults to 'source_estimate_LCMV.h5' for backward compatibility.
        Use 'source_estimate_LCMV_epoch_XXX.h5' for epoch-based extraction.
    stn_radius_mm : float
        Reserved for future use. Currently unused since STN is embedded
        in the unified atlas. Kept for API compatibility.
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
    fsaverage_dir = Path(fsaverage_dir)

    subject_name = subject_output_dir.name
    log = _setup_logger(f'{subject_name}_cimt', subject_output_dir, verbose)

    log.info("=" * 60)
    log.info(f"CIMT Unified Atlas Extraction (448 ROIs): {subject_name}")
    log.info("=" * 60)

    # 1. Validate Inputs
    stc_file = subject_output_dir / stc_filename
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

    # 3. Locate Bundled Unified Atlas
    package_dir = Path(lcmv_xtra.__file__).parent
    atlas_path = package_dir / 'data' / 'cimt_atlas' / 'CIMT_448ROIs_atlas.nii.gz'
    labels_path = package_dir / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Unified CIMT atlas not found: {atlas_path}\n"
            f"Expected: lcmv_xtra/data/cimt_atlas/CIMT_448ROIs_atlas.nii.gz"
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"CIMT labels not found: {labels_path}")

    # 4. Load Atlas & Map Source Vertices to ROI Labels
    log.info("Loading unified CIMT atlas (448 ROIs)...")
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(np.int32)

    src_coords_mni = _get_mni_coordinates(stc, src)
    vox_coords = _coords_to_voxels(src_coords_mni, atlas_img)
    valid_mask = _filter_valid_voxels(vox_coords, atlas_data.shape)
    valid_indices = np.where(valid_mask)[0]
    valid_voxels = vox_coords[valid_mask]

    log.info(f"Mapped {len(valid_indices)} / {len(src_coords_mni)} sources to atlas voxels")

    # Assign each source vertex to its ROI label
    vertex_labels = np.zeros(len(src_coords_mni), dtype=np.int32)
    for i, (x, y, z) in enumerate(valid_voxels):
        vertex_labels[valid_indices[i]] = atlas_data[x, y, z]

    # 5. Extract Time Courses Per ROI
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    n_rois = 448
    time_courses = np.full((n_rois, n_times), np.nan, dtype=np.float32)

    log.info("Extracting time courses for 448 ROIs...")
    for roi_label in range(1, n_rois + 1):
        mask = vertex_labels == roi_label
        if np.any(mask):
            time_courses[roi_label - 1, :] = np.mean(stc_data[mask, :], axis=0)

    time_courses = np.nan_to_num(time_courses, nan=0.0)

    # 6. Load & Validate Labels
    cimt_labels = pd.read_csv(labels_path)

    assert time_courses.shape[0] == 448, f"Row count mismatch: {time_courses.shape[0]}"
    assert len(cimt_labels) == 448, f"Label count mismatch: {len(cimt_labels)}"
    assert list(cimt_labels['index']) == list(range(448)), "Indices not sequential!"

    # Zero-variance ROI quality check
    zero_var_mask = np.std(time_courses, axis=1) < 1e-12
    n_zero_var = int(zero_var_mask.sum())
    if n_zero_var > 0:
        zero_var_names = cimt_labels.loc[zero_var_mask, 'roi_name'].tolist()
        log.warning(
            f"{n_zero_var}/448 ROIs have zero variance "
            f"(no valid sources assigned or silent region):"
        )
        for name in zero_var_names[:10]:
            log.warning(f"   - {name}")
        if n_zero_var > 10:
            log.warning(f"   ... and {n_zero_var - 10} more")
    else:
        log.info("All 448 ROIs have non-zero variance")

    # 7. Save Outputs
    out_file = subject_output_dir / "cimt_time_courses.npy"
    np.save(out_file, time_courses)
    cimt_labels.to_csv(subject_output_dir / "cimt_atlas_labels.csv", index=False)

    log.info("CIMT Extraction Complete!")
    log.info(f"   Shape: {time_courses.shape}")
    log.info("=" * 60 + "\n")

    return time_courses, cimt_labels
