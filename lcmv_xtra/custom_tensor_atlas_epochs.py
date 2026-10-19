# lcmv_xtra/custom_tensor_epochs_atlas.py
"""
Custom Atlas-Constrained Epoch Tensor Assembly.
================================================
Extracts time courses from user-defined MNI coordinates (e.g., DBS contacts)
by mapping them to the nearest available CIMT atlas ROIs.

Because the source estimation uses the Atlas-Before-Inverse approach, the 
beamformer outputs exactly 448 ROI time courses. This script maps custom 
coordinates to their closest atlas index and extracts only those specific 
time courses, producing a 4D tensor: (Subjects, Custom_ROIs, Epochs, Time).
"""

import os
import logging
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .source_estimation_epochs_atlas import execute_source_estimation_epochs_atlas
from .source_estimation_atlas import lookup_mni_coordinate

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SFREQ = 250.0
N_CIMT_ROIS = 448


def map_custom_coords_to_atlas(
    roi_coordinates: Dict[str, List[float]], 
    radius_mm: float = 5.0
) -> Tuple[List[int], List[str]]:
    """
    Maps user-defined MNI coordinates to the nearest CIMT atlas ROI indices.
    
    Parameters
    ----------
    roi_coordinates : dict
        Mapping of ROI name -> [x, y, z] in MNI mm.
    radius_mm : float
        Search radius for the atlas lookup.
        
    Returns
    -------
    roi_indices : list of int
        The 0-based index (0-447) in the atlas STC for each custom ROI.
    roi_names : list of str
        The user-provided names for each custom ROI.
    """
    roi_indices = []
    roi_names = []
    
    logger.info("")
    logger.info(f"{'ROI Name':<18} | {'Index':<6} | {'Dist (mm)':<10} | Atlas Region")
    logger.info("-" * 85)
    
    for roi_name, coord in roi_coordinates.items():
        res = lookup_mni_coordinate(coord, radius_mm=radius_mm)
        idx = res['roi_index']
        dist = res['distance_mm']
        region = res['region_full_name']
        
        logger.info(f"{roi_name:<18} | {idx:<6} | {dist:<10.2f} | {region}")
        
        roi_indices.append(idx)
        roi_names.append(roi_name)
        
    return roi_indices, roi_names


def _process_single_subject_custom_atlas_epochs(args: Tuple) -> Dict:
    """
    Process one subject: epoch atlas source estimation + direct extraction 
    of the mapped custom ROI indices.
    """
    (
        sid, epochs_path, task_name, project_base, fs_dir,
        verbose, reg,
        noise_cov_method, baseline_tmin, baseline_tmax,
        data_cov_tmin, data_cov_tmax, compute_noise_cov,
        roi_indices, custom_roi_names
    ) = args

    try:
        # 1. Run epoch-based atlas source estimation (yields 448 ROIs)
        metadata = execute_source_estimation_epochs_atlas(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            epochs_file_path=epochs_path,
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

        # 2. Extract custom ROI time courses PER EPOCH
        epoch_time_courses = []
        for i in range(n_epochs):
            stc_file = subject_output / f'source_estimate_LCMV_epoch_{i:03d}.h5'
            if not stc_file.exists():
                logger.warning(f"Missing epoch STC for {sid} epoch {i}, skipping")
                continue

            stc = mne.read_source_estimate(str(stc_file))
            
            # Verify shape matches the atlas contract
            if stc.data.shape[0] != N_CIMT_ROIS:
                raise ValueError(f"Expected {N_CIMT_ROIS} ROIs, got {stc.data.shape[0]} for {sid} epoch {i}")
                
            # Extract ONLY the requested custom ROI indices directly from the 448-ROI STC
            # Shape: (n_custom_rois, time)
            tc = stc.data[roi_indices, :].astype(np.float32)  
            epoch_time_courses.append(tc)

        if not epoch_time_courses:
            raise RuntimeError(f"No valid epoch STCs found for {sid}")

        # Stack to (n_epochs, n_custom_rois, time)
        stacked_epochs = np.stack(epoch_time_courses, axis=0)

        return {
            "subject_id": sid,
            "epoch_time_courses": stacked_epochs, 
            "roi_names": custom_roi_names,
            "sfreq": metadata['sfreq_hz'],
            "n_epochs": len(epoch_time_courses),
            "coreg_error_mm": metadata['coreg_mean_error_mm'],
            "success": True,
        }
    except Exception as e:
        logger.error(f"Failed {sid}: {e}", exc_info=True)
        return {"subject_id": sid, "success": False, "error": str(e)}


def save_custom_atlas_tensor_epochs(
    all_subject_data: List[Dict],
    task_name: str,
    output_dir: Path,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> Path:
    """Stack into 4D array: (Subjects, Custom_ROIs, Epochs, Time_per_epoch)."""
    if not all_subject_data:
        raise ValueError("No data to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_data = []
    subject_ids = []

    for d in all_subject_data:
        etc = d['epoch_time_courses']  # (n_epochs, n_custom_rois, time)
        current_sfreq = d['sfreq']

        # Resample time axis (axis=2)
        if abs(current_sfreq - target_sfreq) > 0.5:
            up = int(target_sfreq)
            down = int(round(current_sfreq))
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
    ])  # Shape: (n_subjects, min_epochs, n_custom_rois, min_time)

    # Reorder to (Subjects, ROIs, Epochs, Time)
    stacked = np.transpose(stacked, (0, 2, 1, 3))

    roi_names = all_subject_data[0]['roi_names']

    output_path = output_dir / f"tensor_{task_name}_custom_atlas_epochs.npz"
    np.savez_compressed(
        output_path,
        data=stacked,
        subject_ids=np.array(subject_ids),
        roi_names=np.array(roi_names),
        sfreq=target_sfreq,
        n_epochs=min_epochs,
        n_rois=stacked.shape[1],
        atlas="CIMT_448_Mapped",
    )

    logger.info(
        f"Saved 4D Custom Atlas Epoch Tensor {stacked.shape} "
        f"(Subjects, ROIs, Epochs, Time) at {target_sfreq} Hz to {output_path}"
    )
    return output_path


def assemble_custom_atlas_tensor_epochs(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    roi_coordinates: Dict[str, List[float]],
    task_name: str = "study",
    project_base: Optional[Path] = None,
    search_radius_mm: float = 5.0,
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
    Assemble a 4D custom ROI epoch tensor using the Atlas-Before-Inverse approach.
    
    Maps user-defined MNI coordinates (e.g., DBS contacts) to the nearest 
    available atlas ROIs, runs the 448-ROI beamformer, and extracts only those 
    specific ROI time courses.
    
    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'epochs_path' columns. 
    roi_coordinates : dict
        Mapping of ROI name → [x, y, z] MNI coordinates (mm).
        Example: {"R1": [10.958, 5.563, -4.948], "L1": [-15.983, 6.077, 0.027]}
    search_radius_mm : float
        Maximum distance to accept an atlas ROI as a match for the custom coordinate.
    """
    if data_index.empty:
        return None
    if project_base is None:
        project_base = Path.cwd()

    # Map coordinates to atlas ROIs ONCE in the main process before spinning up workers
    roi_indices, custom_roi_names = map_custom_coords_to_atlas(
        roi_coordinates, radius_mm=search_radius_mm
    )

    tasks = [
        (
            row['subject_id'], row['epochs_path'], task_name,
            project_base, fs_dir, verbose, reg,
            noise_cov_method, baseline_tmin, baseline_tmax,
            data_cov_tmin, data_cov_tmax, compute_noise_cov,
            roi_indices, custom_roi_names,
        )
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers...")
    logger.info(f"    Backend: CIMT atlas-before-inverse mapped to {len(custom_roi_names)} custom ROIs")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject_custom_atlas_epochs, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)
                logger.info(
                    f"  ✅ {result['subject_id']}: "
                    f"{result['epoch_time_courses'].shape} @ {result['sfreq']}Hz "
                    f"(coreg={result['coreg_error_mm']:.2f}mm)"
                )
            else:
                logger.error(f"  ❌ {result['subject_id']}: {result.get('error', 'Unknown error')}")

    if all_subject_data:
        return save_custom_atlas_tensor_epochs(
            all_subject_data, task_name, output_dir, target_sfreq=target_sfreq,
        )
        
    logger.error("All subjects failed. No tensor produced.")
    return None
