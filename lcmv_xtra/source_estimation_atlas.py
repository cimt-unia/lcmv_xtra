# lcmv_xtra/source_estimation_atlas.py
"""
CIMT-Constrained Source Estimation (Atlas-Before-Inverse)
==========================================================
LCMV beamformer with CIMT-reduced source space (448 ROIs).

The forward model is reduced to 448 ROI columns BEFORE make_lcmv(),
so the beamformer operates directly at the atlas level. This eliminates
post-hoc cimt_extraction() and prevents intra-ROI signal cancellation.

Output: stc.data shape (448, T) — directly compatible with
        compute_cimt_motor_connectivity() and assemble_tensor().
"""

import mne
import json
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from nilearn import image

import lcmv_xtra
from lcmv_xtra.atlas_extraction import _get_mni_coordinates, _coords_to_voxels
from lcmv_xtra.source_estimation import (
    load_subject,
    validate_fsaverage,
    _run_coregistration,
    _setup_logger,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ATLAS COORDINATE LOOKUP UTILITIES
# =============================================================================

def _load_cimt_atlas():
    """Load the bundled CIMT atlas and labels CSV."""
    package_dir = Path(lcmv_xtra.__file__).parent
    atlas_path = package_dir / 'data' / 'cimt_atlas' / 'CIMT_448ROIs_atlas.nii.gz'
    labels_path = package_dir / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"CIMT atlas not found: {atlas_path}\n"
            f"Expected: lcmv_xtra/data/cimt_atlas/CIMT_448ROIs_atlas.nii.gz"
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"CIMT labels not found: {labels_path}")

    import pandas as pd
    atlas_img = nib.load(str(atlas_path))
    labels_df = pd.read_csv(labels_path)
    return atlas_img, labels_df


def lookup_mni_coordinate(
    mni_coord: Union[List[float], np.ndarray],
    radius_mm: float = 5.0,
) -> Dict:
    """
    Find the closest CIMT ROI to an arbitrary MNI coordinate.

    Parameters
    ----------
    mni_coord : list or array of shape (3,)
        MNI coordinates in mm, e.g., [-11.89, -14.51, -6.40].
    radius_mm : float
        Search radius in mm. Falls back to nearest labeled voxel
        if nothing is found within this radius.

    Returns
    -------
    dict with keys:
        'roi_index'       : int (0-447)
        'nifti_label'     : int (1-448)
        'roi_name'        : str (e.g., 'STN-lh')
        'region_full_name': str (e.g., 'Subthalamic Nucleus')
        'hemisphere'      : str ('L', 'R', or 'B')
        'functional_system': str
        'distance_mm'     : float
        'voxel_mni'       : list of 3 floats
    """
    atlas_img, labels_df = _load_cimt_atlas()
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    coord = np.asarray(mni_coord, dtype=np.float64)

    # Convert MNI → voxel index
    inv_affine = np.linalg.inv(atlas_img.affine)
    vox = np.round(inv_affine @ np.append(coord, 1.0)).astype(int)[:3]

    shape = atlas_data.shape
    in_bounds = np.all(vox >= 0) and np.all(vox < shape)

    if in_bounds:
        label = int(atlas_data[vox[0], vox[1], vox[2]])
        vox_mni = nib.affines.apply_affine(atlas_img.affine, vox.astype(float))
        distance_mm = float(np.linalg.norm(vox_mni - coord))

        # If unlabeled or outside radius, search neighborhood
        if label == 0 or distance_mm > radius_mm:
            r_vox = int(np.ceil(radius_mm / np.min(np.abs(np.diag(atlas_img.affine)[:3])))) + 1
            x_min, x_max = max(0, vox[0] - r_vox), min(shape[0], vox[0] + r_vox + 1)
            y_min, y_max = max(0, vox[1] - r_vox), min(shape[1], vox[1] + r_vox + 1)
            z_min, z_max = max(0, vox[2] - r_vox), min(shape[2], vox[2] + r_vox + 1)

            sub_grid = atlas_data[x_min:x_max, y_min:y_max, z_min:z_max]
            labeled_local = np.argwhere(sub_grid > 0)

            if len(labeled_local) > 0:
                labeled_global = labeled_local + np.array([x_min, y_min, z_min])
                labeled_mni = nib.affines.apply_affine(atlas_img.affine, labeled_global.astype(float))
                distances = np.linalg.norm(labeled_mni - coord, axis=1)
                nearest = np.argmin(distances)
                vox = labeled_global[nearest]
                label = int(atlas_data[vox[0], vox[1], vox[2]])
                vox_mni = labeled_mni[nearest]
                distance_mm = float(distances[nearest])
            else:
                # Global fallback
                labeled_voxels = np.argwhere(atlas_data > 0)
                labeled_mni = nib.affines.apply_affine(atlas_img.affine, labeled_voxels.astype(float))
                distances = np.linalg.norm(labeled_mni - coord, axis=1)
                nearest = np.argmin(distances)
                vox = labeled_voxels[nearest]
                label = int(atlas_data[vox[0], vox[1], vox[2]])
                vox_mni = labeled_mni[nearest]
                distance_mm = float(distances[nearest])
    else:
        # Outside atlas volume entirely — global search
        labeled_voxels = np.argwhere(atlas_data > 0)
        labeled_mni = nib.affines.apply_affine(atlas_img.affine, labeled_voxels.astype(float))
        distances = np.linalg.norm(labeled_mni - coord, axis=1)
        nearest = np.argmin(distances)
        vox = labeled_voxels[nearest]
        label = int(atlas_data[vox[0], vox[1], vox[2]])
        vox_mni = labeled_mni[nearest]
        distance_mm = float(distances[nearest])

    roi_index = label - 1  # NIfTI 1-448 → CSV index 0-447
    row = labels_df.iloc[roi_index]

    return {
        'roi_index': roi_index,
        'nifti_label': label,
        'roi_name': str(row['roi_name']),
        'region_full_name': str(row['region_full_name']),
        'hemisphere': str(row['hemisphere']),
        'functional_system': str(row['functional_system']),
        'distance_mm': round(distance_mm, 2),
        'voxel_mni': [round(float(v), 2) for v in vox_mni],
    }


def lookup_multiple_coordinates(
    coordinates: List[List[float]],
    radius_mm: float = 5.0,
) -> List[Dict]:
    """Batch lookup for multiple MNI coordinates."""
    return [lookup_mni_coordinate(c, radius_mm) for c in coordinates]


# =============================================================================
# LEAD FIELD REDUCTION
# =============================================================================

def reduce_leadfield_to_cimt(fwd, src, verbose=False):
    """
    Reduce full volumetric lead field to 448 CIMT ROI channels.

    Uses the pre-built CIMT_448ROIs_atlas.nii.gz bundled in the package.
    Each source voxel is assigned to its atlas ROI via MNI coordinate lookup.
    Lead field columns are then averaged within each ROI.

    Parameters
    ----------
    fwd : mne.Forward
        Full forward solution (from mne.make_forward_solution).
    src : mne.SourceSpaces
        Volume source space (fsaverage-vol-5mm-src.fif).
    verbose : bool
        Enable logging.

    Returns
    -------
    G_reduced : np.ndarray, shape (n_channels, 448)
    voxel_labels : np.ndarray, shape (n_sources,), values 0-447 or -1
    roi_counts : np.ndarray, shape (448,)
    """
    log = logger if verbose else logging.getLogger(__name__)

    atlas_img, _ = _load_cimt_atlas()
    atlas_data = atlas_img.get_fdata().astype(np.int32)

    # Get MNI coordinates of all source voxels
    n_sources = len(src[0]['vertno'])
    src_rr = src[0]['rr'][src[0]['vertno']] * 1000.0  # m → mm

    try:
        trans = src[0]['mri_ras_t']['trans']
    except KeyError:
        raise ValueError(
            "Source space missing 'mri_ras_t' transform. "
            "Ensure it is a proper volume source space."
        )

    mni_coords = np.array(
        image.coord_transform(src_rr[:, 0], src_rr[:, 1], src_rr[:, 2], trans)
    ).T  # (n_sources, 3)

    # Look up atlas label for each source voxel
    vox_coords = _coords_to_voxels(mni_coords, atlas_img)
    voxel_labels = np.full(n_sources, -1, dtype=np.int64)

    shape = atlas_data.shape
    for i, (x, y, z) in enumerate(vox_coords):
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            label = atlas_data[x, y, z]
            if label > 0:
                voxel_labels[i] = label - 1

    n_assigned = np.sum(voxel_labels >= 0)
    if verbose:
        log.info(f"CIMT reduction: {n_assigned}/{n_sources} voxels assigned")
        log.info(f"  {n_sources - n_assigned} voxels outside atlas (excluded)")

    # Reduce lead field
    G_full = fwd['sol']['data']
    n_channels = G_full.shape[0]
    n_dipoles = G_full.shape[1]

    if n_dipoles == n_sources * 3:
        has_free_ori = True
    elif n_dipoles == n_sources:
        has_free_ori = False
    else:
        raise ValueError(
            f"Unexpected lead field shape: ({n_channels}, {n_dipoles}) "
            f"for {n_sources} sources."
        )

    G_reduced = np.zeros((n_channels, 448), dtype=np.float32)
    roi_counts = np.zeros(448, dtype=np.int32)

    for k in range(448):
        mask = (voxel_labels == k)
        n_vox = np.sum(mask)
        roi_counts[k] = n_vox

        if n_vox == 0:
            continue

        if has_free_ori:
            voxel_indices = np.where(mask)[0]
            dipole_indices = np.concatenate([v * 3 + np.arange(3) for v in voxel_indices])
            G_roi = G_full[:, dipole_indices]
            G_roi_reshaped = G_roi.reshape(n_channels, n_vox, 3)
            voxel_norms = np.linalg.norm(G_roi_reshaped, axis=2)
            G_reduced[:, k] = voxel_norms.mean(axis=1)
        else:
            G_reduced[:, k] = G_full[:, mask].mean(axis=1)

    if verbose:
        empty_rois = np.sum(roi_counts == 0)
        log.info(f"Reduced lead field: ({n_channels}, 448)")
        if empty_rois > 0:
            log.warning(f"  {empty_rois} ROIs have no source voxels assigned")

    return G_reduced, voxel_labels, roi_counts


# =============================================================================
# CIMT-CONSTRAINED LCMV BEAMFORMER
# =============================================================================

def lcmv_beamformer_cimt(
    input,
    ch_pos,
    fsaverage_dir,
    output_dir,
    subject_id,
    task,
    reg=0.05,
    n_jobs=1,
    verbose=False,
):
    """
    LCMV beamformer with CIMT-reduced source space (448 ROIs).

    Identical to lcmv_beamformer() except the forward model is reduced
    to 448 ROI columns before make_lcmv(). Output stc.data is (448, T).

    Returns
    -------
    metadata : dict
        Same structure as lcmv_beamformer(), with 'source_space': 'CIMT_448_ROIs'.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info(f"{'=' * 60}")
    log.info(f"LCMV-CIMT Source Estimation: {subject_id} - {task}")
    log.info(f"{'=' * 60}")

    # Validate resources
    bem_file, src_file = validate_fsaverage(fsaverage_dir)

    # Coregistration (unchanged)
    log.info("Running coregistration...")
    trans_file = output_dir / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(
        input, ch_pos, 'fsaverage', fsaverage_dir, trans_file, log
    )

    # Source space
    log.info("Loading source space...")
    src = mne.read_source_spaces(src_file)
    n_active = len(src[0]['vertno'])
    log.info(f"Source space: {n_active} active sources")

    # Forward solution (full, unchanged)
    log.info("Computing forward solution...")
    fwd_file = output_dir / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        input.info, trans=trans, src=src, bem=bem,
        eeg=True, mindist=5.0, n_jobs=n_jobs,
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    # ── Reduce to 448 CIMT ROIs ──
    log.info("Reducing lead field to 448 CIMT ROIs...")
    G_reduced, voxel_labels, roi_counts = reduce_leadfield_to_cimt(
        fwd=fwd, src=src, verbose=True,
    )

    # ── Update ALL forward solution metadata to match reduced dimensions ──
    # MNE's _compute_beamformer asserts nn.shape == (n_sources, 3) internally.
    # Without updating source_nn and vertno, make_lcmv raises AssertionError.
    n_roi = 448
    fwd['sol']['data'] = G_reduced
    fwd['nsource'] = n_roi
    fwd['sol']['source_nn'] = np.tile([0.0, 0.0, 1.0], (n_roi, 1)).astype(np.float32)
    fwd['src'][0]['vertno'] = np.arange(n_roi, dtype=np.int32)

    # Save reduced lead field and voxel mapping
    np.save(output_dir / 'G_cimt_448.npy', G_reduced)
    np.save(output_dir / 'cimt_voxel_labels.npy', voxel_labels)

    # LCMV beamformer (operates on 448 ROIs)
    log.info("Computing covariance and LCMV filters (448 ROIs)...")
    cov = mne.compute_raw_covariance(
        input, method='oas', picks='eeg', rank=None, n_jobs=n_jobs, verbose=False,
    )
    filters = mne.beamformer.make_lcmv(
        info=input.info, forward=fwd, data_cov=cov, noise_cov=cov, reg=reg,
        pick_ori='max-power', weight_norm='unit-noise-gain',
        reduce_rank=True, rank=None, verbose=False,
    )

    log.info("Applying LCMV beamformer to data...")
    stc = mne.beamformer.apply_lcmv_raw(raw=input, filters=filters)

    stc_file = output_dir / 'source_estimate_LCMV.h5'
    stc.save(stc_file, ftype='h5', overwrite=True)
    log.info(f"Saved CIMT source estimate: {stc.data.shape[0]} ROIs × {stc.data.shape[1]} timepoints")

    # Metadata
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'sfreq_hz': float(input.info['sfreq']),
        'duration_min': float(input.n_times / input.info['sfreq'] / 60),
        'n_sources': 448,
        'n_timepoints': int(stc.data.shape[1]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'source_space': 'CIMT_448_ROIs',
        'subject_output': str(output_dir),
        'fsaverage_dir': str(fsaverage_dir),
    }
    with open(output_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"LCMV-CIMT source estimation complete: {output_dir}")
    log.info(f"{'=' * 60}\n")

    return metadata


# =============================================================================
# HIGH-LEVEL ORCHESTRATOR
# =============================================================================

def execute_source_estimation_atlas(
    project_base,
    subject_id,
    task,
    ica_file_path,
    fsaverage_dir,
    reg=0.05,
    n_jobs=1,
    verbose=False,
):
    """
    High-level orchestrator for CIMT-constrained source estimation.

    Drop-in replacement for execute_source_estimation(). Produces
    (448, T) output directly — no cimt_extraction() needed afterward.

    Parameters
    ----------
    project_base : str or Path
        Root of BIDS-like project.
    subject_id : str
        Subject identifier (e.g., 'sub-01').
    task : str
        Task name (e.g., 'rest').
    ica_file_path : str
        Relative path to cleaned .fif within project_base.
    fsaverage_dir : str or Path
        Directory containing fsaverage/ and fsaverage-vol-5mm-src.fif.
    reg : float
        LCMV regularization parameter.
    n_jobs : int
        Parallel jobs for forward solution.
    verbose : bool
        Enable console logging.

    Returns
    -------
    metadata : dict
    """
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'

    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")

    ica_full_path = project_base / ica_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_cimt'

    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path,
        gpsc_file_path=gpsc_full_path,
        subject_id=subject_id,
        logger=None,
    )

    return lcmv_beamformer_cimt(
        input=raw,
        ch_pos=ch_pos,
        fsaverage_dir=fsaverage_dir,
        output_dir=output_dir,
        subject_id=subject_id,
        task=task,
        reg=reg,
        n_jobs=n_jobs,
        verbose=verbose,
    )
