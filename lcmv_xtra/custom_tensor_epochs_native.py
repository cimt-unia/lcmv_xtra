# lcmv_xtra/custom_tensor_epochs_native.py
"""
Custom Voxel Epoch Tensor Assembly (Native Space Forward Model).
=====================================================================
Extracts time courses from user-defined coordinates per epoch,
producing a 4D tensor: (Subjects, ROIs, Epochs, Time_per_epoch).

Supports two modes via `morph_to_fsaverage` flag:
1. morph_to_fsaverage=False (Default): 
   Extracts directly from un-morphed native STCs. Coordinates MUST be 
   in the subject's Native Surface RAS space (mm). Use the provided 
   `transform_clinical_to_native_source_space` utility to convert 
   clinical/DICOM coordinates to Native Surface RAS.
2. morph_to_fsaverage=True: 
   Morphs STCs to fsaverage and extracts using standard MNI coordinates.
"""

import os
import logging
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from typing import List, Dict, Optional, Tuple, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

from .source_estimation_epochs_native import execute_source_estimation_epochs_native

logger = logging.getLogger(__name__)

METERS_TO_MILLIMETERS: float = 1000.0
DEFAULT_RADIUS_MM: float = 5.0
DEFAULT_TARGET_SFREQ: float = 250.0

# =============================================================================
# COORDINATE TRANSFORMATION UTILITIES
# =============================================================================
def transform_clinical_to_native_source_space(
    clinical_coords_mm: np.ndarray,
    native_src_file: Path,
    coord_frame: str = 'NIfTI_RAS'
) -> np.ndarray:
    """
    Transforms clinical LFP coordinates into FreeSurfer Surface RAS (mm)
    for direct extraction from native MNE source estimates.
    
    Parameters
    ----------
    clinical_coords_mm : array-like, shape (3,) or (N, 3)
        The [x, y, z] coordinates from your surgical planning software.
    native_src_file : Path
        Path to the subject's native volume source space.
    coord_frame : str
        'NIfTI_RAS' (Right, Anterior, Superior) or 'DICOM_LPS' (Left, Posterior, Superior).
    """
    coords = np.atleast_2d(clinical_coords_mm).astype(float)
    
    if coord_frame == 'DICOM_LPS':
        coords[:, 0] = -coords[:, 0]  # Left -> Right
        coords[:, 1] = -coords[:, 1]  # Posterior -> Anterior
        
    coords_m = coords / 1000.0
    src = mne.read_source_spaces(native_src_file)
    
    try:
        mri_ras_t = src[0]['mri_ras_t']['trans']
    except KeyError:
        raise ValueError("Source space missing 'mri_ras_t' transform.")
        
    ras_to_surf = np.linalg.inv(mri_ras_t)
    coords_homog = np.hstack([coords_m, np.ones((coords_m.shape[0], 1))])
    surf_ras_m = (ras_to_surf @ coords_homog.T).T[:, :3]
    
    return surf_ras_m * 1000.0

# =============================================================================
# ROI EXTRACTION
# =============================================================================
def extract_custom_roi_time_courses(
    stc: mne.SourceEstimate,
    src: mne.SourceSpaces,
    roi_coordinates: Dict[str, List[float]],
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract time courses from user-defined coordinates.
    Works for both Native Surface RAS (if src is native) and MNI (if src is fsaverage).
    """
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    
    vertices = stc.vertices[0]
    active_coords_mm = src[0]['rr'][vertices] * METERS_TO_MILLIMETERS

    time_courses, roi_names = [], []

    for roi_name, target_coord in roi_coordinates.items():
        target = np.array(target_coord, dtype=np.float64)
        distances = np.linalg.norm(active_coords_mm - target, axis=1)

        if mode == "single":
            selected_indices = np.array([np.argmin(distances)])
        else:  # sphere
            selected_indices = np.where(distances <= radius_mm)[0]
            if selected_indices.size == 0:
                logger.warning(
                    f"No vertices within {radius_mm}mm of {roi_name} {target_coord}. "
                    f"Using nearest vertex at {distances.min():.2f}mm."
                )
                selected_indices = np.array([np.argmin(distances)])

        roi_data = stc_data[selected_indices, :].mean(axis=0).astype(np.float32)
        time_courses.append(roi_data)
        roi_names.append(roi_name)

    return np.vstack(time_courses), roi_names

# =============================================================================
# SINGLE SUBJECT PROCESSING
# =============================================================================
def _process_single_subject_custom_epochs_native(args: Tuple) -> Dict:
    """Process one subject: native epoch source estimation + per-epoch ROI extraction."""
    (
        sid, epochs_path, task_name, project_base, subjects_dir, fs_dir,
        verbose, reg,
        noise_cov_method, baseline_tmin, baseline_tmax,
        data_cov_tmin, data_cov_tmax, compute_noise_cov,
        roi_coordinates, radius_mm, mode, coregister, morph_to_fsaverage
    ) = args

    try:
        # 1. Run NATIVE epoch-based source estimation
        metadata = execute_source_estimation_epochs_native(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            epochs_file_path=epochs_path,
            subjects_dir=subjects_dir,
            fsaverage_dir=fs_dir,
            reg=reg,
            verbose=verbose,
            noise_cov_method=noise_cov_method,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
            data_cov_tmin=data_cov_tmin,
            data_cov_tmax=data_cov_tmax,
            compute_noise_cov=compute_noise_cov,
            coregister=coregister,
            morph_to_fsaverage=morph_to_fsaverage,
        )

        subject_output = Path(metadata['subject_output'])
        n_epochs = metadata['n_epochs']
        
        # 2. Determine Source Space and STC paths based on morph flag
        if morph_to_fsaverage:
            src_file = Path(fs_dir) / "fsaverage-vol-5mm-src.fif"
            stc_dir = subject_output
            stc_pattern = "source_estimate_LCMV_epoch_{i:03d}.h5"
            logger.info(f"[{sid}] Extracting from MORPHED fsaverage STCs (MNI coordinates).")
        else:
            src_file = Path(subjects_dir) / sid / f"{sid}-vol-5mm-src.fif"
            stc_dir = subject_output / "native_stcs"
            stc_pattern = "source_estimate_LCMV_epoch_{i:03d}_native.h5"
            logger.info(f"[{sid}] Extracting from PURE NATIVE STCs (Native Surface RAS coordinates).")
            
        src = mne.read_source_spaces(str(src_file))

        # 3. Extract custom ROI time courses PER EPOCH
        epoch_time_courses = []
        for i in range(n_epochs):
            stc_file = stc_dir / stc_pattern.format(i=i)
            if not stc_file.exists():
                logger.warning(f"Missing STC for {sid} epoch {i} at {stc_file}, skipping")
                continue

            stc = mne.read_source_estimate(str(stc_file))
            tc, _ = extract_custom_roi_time_courses(
                stc=stc, src=src,
                roi_coordinates=roi_coordinates,
                radius_mm=radius_mm, mode=mode,
            )
            epoch_time_courses.append(tc)

        if not epoch_time_courses:
            raise RuntimeError(f"No valid STCs found for {sid}")

        return {
            "subject_id": sid,
            "epoch_time_courses": np.stack(epoch_time_courses, axis=0),
            "roi_names": list(roi_coordinates.keys()),
            "sfreq": metadata['sfreq_hz'],
            "n_epochs": len(epoch_time_courses),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Failed {sid}: {e}", exc_info=True)
        return {"subject_id": sid, "success": False}

# =============================================================================
# TENSOR ASSEMBLY
# =============================================================================
def save_custom_tensor_epochs_native(
    all_subject_data: List[Dict],
    task_name: str,
    output_dir: Path,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    morph_to_fsaverage: bool = False,
) -> Path:
    """Stack into 4D array: (Subjects, ROIs, Epochs, Time_per_epoch)."""
    if not all_subject_data:
        raise ValueError("No data to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_data = []
    subject_ids = []

    for d in all_subject_data:
        etc = d['epoch_time_courses']
        current_sfreq = d['sfreq']

        if current_sfreq != target_sfreq:
            up = int(target_sfreq)
            down = int(current_sfreq)
            gcd = np.gcd(up, down)
            etc_resampled = signal.resample_poly(etc, up // gcd, down // gcd, axis=2)
            resampled_data.append(etc_resampled)
        else:
            resampled_data.append(etc)
        subject_ids.append(d['subject_id'])

    min_epochs = min(arr.shape[0] for arr in resampled_data)
    min_time = min(arr.shape[2] for arr in resampled_data)

    stacked = np.stack([
        arr[:min_epochs, :, :min_time] for arr in resampled_data
    ])

    stacked = np.transpose(stacked, (0, 2, 1, 3))
    roi_names = all_subject_data[0]['roi_names']

    space_tag = "morphed_to_fsaverage" if morph_to_fsaverage else "pure_native"
    output_path = output_dir / f"tensor_{task_name}_custom_epochs_native_{space_tag}.npz"
    
    np.savez_compressed(
        output_path,
        data=stacked,
        subject_ids=np.array(subject_ids),
        roi_names=np.array(roi_names),
        sfreq=target_sfreq,
        n_epochs=min_epochs,
        n_rois=stacked.shape[1],
        forward_model=f"native_T1_{space_tag}"
    )

    logger.info(
        f"Saved 4D Native Custom Epoch Tensor {stacked.shape} "
        f"(Subjects, ROIs, Epochs, Time) at {target_sfreq} Hz to {output_path}"
    )
    return output_path


def assemble_custom_tensor_epochs_native(
    data_index: pd.DataFrame,
    subjects_dir: Path,
    fs_dir: Path,
    output_dir: Path,
    roi_coordinates: Dict[str, List[float]],
    task_name: str = "study",
    project_base: Optional[Path] = None,
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
    n_jobs: int = -1,
    verbose: bool = False,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    reg: float = 0.05,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
    coregister: bool = True,
    morph_to_fsaverage: bool = False,
) -> Optional[Path]:
    """
    Assemble a 4D custom ROI epoch tensor using the SUBJECT'S NATIVE T1.
    
    Parameters
    ----------
    morph_to_fsaverage : bool
        If False (default), extracts directly from un-morphed native STCs. 
        `roi_coordinates` MUST be in the subject's Native Surface RAS space (mm).
        If True, morphs STCs to fsaverage and extracts using MNI coordinates.
    """
    if data_index.empty:
        return None
    if project_base is None:
        project_base = Path.cwd()

    tasks = [
        (
            row['subject_id'], row['epochs_path'], task_name,
            project_base, subjects_dir, fs_dir, verbose, reg,
            noise_cov_method, baseline_tmin, baseline_tmax,
            data_cov_tmin, data_cov_tmax, compute_noise_cov,
            roi_coordinates, radius_mm, mode, coregister, morph_to_fsaverage
        )
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    space_str = "MNI (fsaverage)" if morph_to_fsaverage else "Native Surface RAS"
    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers (Native Forward Model)...")
    logger.info(f"Extraction Space: {space_str}")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject_custom_epochs_native, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)

    if all_subject_data:
        return save_custom_tensor_epochs_native(
            all_subject_data, task_name, output_dir, 
            target_sfreq=target_sfreq,
            morph_to_fsaverage=morph_to_fsaverage
        )
    return None
