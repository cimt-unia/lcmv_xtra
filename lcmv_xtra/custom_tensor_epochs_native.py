# lcmv_xtra/custom_tensor_epochs_native.py
"""
Custom MNI Voxel Epoch Tensor Assembly (Native Space Forward Model).
=====================================================================
Extracts time courses from user-defined MNI coordinates per epoch,
producing a 4D tensor: (Subjects, ROIs, Epochs, Time_per_epoch).

This module uses the native-space forward model (subject's own T1/BEM)
for biophysical accuracy, but relies on the fact that the native pipeline
morphs the resulting STCs to fsaverage space, allowing standard MNI 
coordinate-based ROI extraction.
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

# Import the NATIVE epoch source estimation function
from .source_estimation_epochs_native import execute_source_estimation_epochs_native

logger = logging.getLogger(__name__)

METERS_TO_MILLIMETERS: float = 1000.0
DEFAULT_RADIUS_MM: float = 5.0
DEFAULT_TARGET_SFREQ: float = 250.0


def _get_mni_coordinates_from_src(
    stc: mne.SourceEstimate, 
    src: mne.SourceSpaces
) -> np.ndarray:
    """Convert active source space vertex coordinates to MNI space (mm)."""
    vertices = stc.vertices[0]
    src_rr = src[0]['rr'][vertices] * METERS_TO_MILLIMETERS
    
    # Because the STC was morphed to fsaverage, these coordinates are 
    # effectively MNI-aligned via the fsaverage mri_ras_t transform.
    return src_rr


def extract_custom_roi_time_courses(
    stc: mne.SourceEstimate,
    src: mne.SourceSpaces,
    roi_coordinates: Dict[str, List[float]],
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract time courses from user-defined MNI coordinates.
    (Identical to standard pipeline since STCs are morphed to fsaverage).
    """
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    active_coords_mni = _get_mni_coordinates_from_src(stc, src)

    time_courses, roi_names = [], []

    for roi_name, target_mni in roi_coordinates.items():
        target = np.array(target_mni, dtype=np.float64)
        distances = np.linalg.norm(active_coords_mni - target, axis=1)

        if mode == "single":
            selected_indices = np.array([np.argmin(distances)])
        else:  # sphere
            selected_indices = np.where(distances <= radius_mm)[0]
            if selected_indices.size == 0:
                logger.warning(
                    f"No vertices within {radius_mm}mm of {roi_name} {target_mni}. "
                    f"Using nearest vertex at {distances.min():.2f}mm."
                )
                selected_indices = np.array([np.argmin(distances)])

        roi_data = stc_data[selected_indices, :].mean(axis=0).astype(np.float32)
        time_courses.append(roi_data)
        roi_names.append(roi_name)

    return np.vstack(time_courses), roi_names


def _process_single_subject_custom_epochs_native(args: Tuple) -> Dict:
    """Process one subject: native epoch source estimation + per-epoch custom ROI extraction."""
    (
        sid, epochs_path, task_name, project_base, subjects_dir, fs_dir,
        verbose, reg,
        noise_cov_method, baseline_tmin, baseline_tmax,
        data_cov_tmin, data_cov_tmax, compute_noise_cov,
        roi_coordinates, radius_mm, mode
    ) = args

    try:
        # 1. Run NATIVE epoch-based source estimation
        # This computes the forward model in native space, applies LCMV, 
        # and morphs the resulting STCs to fsaverage space.
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
        )

        subject_output = Path(metadata['subject_output'])
        n_epochs = metadata['n_epochs']
        
        # 2. Load fsaverage source space for MNI coordinate extraction
        # (Since the STCs were morphed to fsaverage, we use the fsaverage grid)
        src_file = Path(fs_dir) / "fsaverage-vol-5mm-src.fif"
        src = mne.read_source_spaces(str(src_file))

        # 3. Extract custom ROI time courses PER EPOCH
        epoch_time_courses = []
        for i in range(n_epochs):
            stc_file = subject_output / f'source_estimate_LCMV_epoch_{i:03d}.h5'
            if not stc_file.exists():
                logger.warning(f"Missing epoch STC for {sid} epoch {i}, skipping")
                continue

            stc = mne.read_source_estimate(str(stc_file))
            tc, _ = extract_custom_roi_time_courses(
                stc=stc, src=src,
                roi_coordinates=roi_coordinates,
                radius_mm=radius_mm, mode=mode,
            )
            epoch_time_courses.append(tc)  # Shape: (n_rois, time_per_epoch)

        if not epoch_time_courses:
            raise RuntimeError(f"No valid epoch STCs found for {sid}")

        return {
            "subject_id": sid,
            "epoch_time_courses": np.stack(epoch_time_courses, axis=0),  # (n_epochs, n_rois, time)
            "roi_names": list(roi_coordinates.keys()),
            "sfreq": metadata['sfreq_hz'],
            "n_epochs": len(epoch_time_courses),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Failed {sid}: {e}", exc_info=True)
        return {"subject_id": sid, "success": False}


def save_custom_tensor_epochs_native(
    all_subject_data: List[Dict],
    task_name: str,
    output_dir: Path,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> Path:
    """Stack into 4D array: (Subjects, ROIs, Epochs, Time_per_epoch)."""
    if not all_subject_data:
        raise ValueError("No data to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_data = []
    subject_ids = []

    for d in all_subject_data:
        etc = d['epoch_time_courses']  # (n_epochs, n_rois, time)
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

    # Truncate to shortest common dimensions
    min_epochs = min(arr.shape[0] for arr in resampled_data)
    min_time = min(arr.shape[2] for arr in resampled_data)

    stacked = np.stack([
        arr[:min_epochs, :, :min_time] for arr in resampled_data
    ])  # (n_subjects, min_epochs, n_rois, min_time)

    # Reorder to (Subjects, ROIs, Epochs, Time)
    stacked = np.transpose(stacked, (0, 2, 1, 3))

    roi_names = all_subject_data[0]['roi_names']

    output_path = output_dir / f"tensor_{task_name}_custom_epochs_native.npz"
    np.savez_compressed(
        output_path,
        data=stacked,
        subject_ids=np.array(subject_ids),
        roi_names=np.array(roi_names),
        sfreq=target_sfreq,
        n_epochs=min_epochs,
        n_rois=stacked.shape[1],
        forward_model="native_T1_morphed_to_fsaverage" # Metadata tag
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
) -> Optional[Path]:
    """
    Assemble a 4D custom ROI epoch tensor using the SUBJECT'S NATIVE T1 
    for the forward model, with results morphed to fsaverage for MNI extraction.
    
    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'epochs_path' columns. 
    subjects_dir : Path
        Directory containing subject-specific FreeSurfer recon-all outputs 
        (e.g., sub-DP02/bem, sub-DP02/mri).
    fs_dir : Path
        Directory containing fsaverage resources (fsaverage-vol-5mm-src.fif).
    roi_coordinates : dict
        Mapping of ROI name → [x, y, z] MNI coordinates (mm).
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
            roi_coordinates, radius_mm, mode,
        )
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers (Native Forward Model)...")

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
            all_subject_data, task_name, output_dir, target_sfreq=target_sfreq,
        )
    return None
