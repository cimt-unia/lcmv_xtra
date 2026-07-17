# lcmv_xtra/custom_tensor.py

import os
import logging
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from typing import Dict, List, Optional, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from nilearn import image

from lcmv_xtra.source_estimation import execute_source_estimation

logger = logging.getLogger(__name__)

METERS_TO_MILLIMETERS: float = 1000.0
DEFAULT_RADIUS_MM: float = 5.0
DEFAULT_TARGET_SFREQ: float = 250.0


def _get_mni_coordinates_from_src(stc, src):
    """Convert source space coordinates to MNI space (matches atlas_extraction.py)."""
    vertices = stc.vertices[0]
    src_rr = src[0]['rr'][vertices] * METERS_TO_MILLIMETERS
    try:
        trans = src[0]['mri_ras_t']['trans']
    except KeyError:
        raise ValueError("Source space missing 'mri_ras_t' transform.")
    mni_coords = image.coord_transform(src_rr[:, 0], src_rr[:, 1], src_rr[:, 2], trans)
    return np.array(mni_coords).T


def extract_custom_roi_time_courses(
    stc: mne.SourceEstimate,
    src: mne.SourceSpaces,
    roi_coordinates: Dict[str, List[float]],
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
    log: Optional[logging.Logger] = None
) -> tuple:
    """Extract time courses from user-defined MNI coordinates."""
    if log is None:
        log = logging.getLogger(__name__)
    
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    active_coords_mni = _get_mni_coordinates_from_src(stc, src)
    
    time_courses, roi_names = [], []
    
    for roi_name, target_mni in roi_coordinates.items():
        target = np.array(target_mni, dtype=np.float64)
        distances = np.linalg.norm(active_coords_mni - target, axis=1)
        
        if mode == "single":
            selected_indices = np.array([np.argmin(distances)])
        else:
            selected_indices = np.where(distances <= radius_mm)[0]
            if selected_indices.size == 0:
                selected_indices = np.array([np.argmin(distances)])
        
        roi_data = stc_data[selected_indices, :].mean(axis=0).astype(np.float32)
        time_courses.append(roi_data)
        roi_names.append(roi_name)
    
    return np.vstack(time_courses), roi_names


def _process_single_subject_mni(args):
    """Helper for parallel processing - MNI coordinate based."""
    sid, fif_path, task_name, project_base, fs_dir, roi_coordinates, radius_mm, mode, verbose = args
    
    try:
        metadata = execute_source_estimation(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=fif_path,
            fsaverage_dir=fs_dir,
            verbose=verbose
        )
        
        stc_file = Path(metadata['subject_output']) / "source_estimate_LCMV.h5"
        src_file = Path(fs_dir) / "fsaverage-vol-5mm-src.fif"
        
        stc = mne.read_source_estimate(str(stc_file))
        src = mne.read_source_spaces(str(src_file))
        
        tc, roi_names = extract_custom_roi_time_courses(
            stc=stc, src=src,
            roi_coordinates=roi_coordinates,
            radius_mm=radius_mm, mode=mode
        )
        
        return {
            "subject_id": sid,
            "time_courses": tc,
            "roi_names": roi_names,
            "sfreq": metadata['sfreq_hz'],
            "success": True
        }
    except Exception as e:
        logger.error(f"Failed {sid}: {e}")
        return {"subject_id": sid, "success": False}


def _resample_time_course_mni(time_course, current_sfreq, target_sfreq):
    """Resample time course to target frequency."""
    if current_sfreq == target_sfreq:
        return time_course
    up, down = int(target_sfreq), int(current_sfreq)
    if up == target_sfreq and down == current_sfreq:
        gcd = np.gcd(up, down)
        return signal.resample_poly(time_course, up // gcd, down // gcd, axis=1)
    duration_sec = time_course.shape[1] / current_sfreq
    n_samples_new = int(np.round(duration_sec * target_sfreq))
    return signal.resample(time_course, n_samples_new, axis=1)


def save_study_tensor_mni(all_subject_data, task_name, output_dir, target_sfreq=DEFAULT_TARGET_SFREQ):
    """Stack into (subjects, ROIs, time) and save."""
    if not all_subject_data:
        raise ValueError("No data to stack.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    resampled_data = [_resample_time_course_mni(d['time_courses'], d['sfreq'], target_sfreq) for d in all_subject_data]
    min_t = min(arr.shape[1] for arr in resampled_data)
    stacked_data = np.stack([arr[:, :min_t] for arr in resampled_data])
    subject_ids = np.array([d['subject_id'] for d in all_subject_data])
    roi_names = all_subject_data[0]['roi_names']
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(output_path, data=stacked_data, subject_ids=subject_ids, roi_names=np.array(roi_names), sfreq=target_sfreq)
    logger.info(f"Saved {stacked_data.shape} at {target_sfreq} Hz to {output_path}")
    return output_path


def assemble_custom_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    roi_coordinates: Dict[str, List[float]],
    task_name: str = "study",
    project_base: Path = None,
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
    n_jobs: int = -1,
    verbose: bool = False,
    target_sfreq: float = DEFAULT_TARGET_SFREQ
) -> Optional[Path]:
    """Source estimation → Custom MNI ROI extraction → 3D Tensor."""
    if data_index.empty:
        return None
    if project_base is None:
        project_base = Path.cwd()
    
    tasks = [
        (row['subject_id'], Path(row['fif_path']), task_name, project_base, fs_dir,
         roi_coordinates, radius_mm, mode, verbose)
        for _, row in data_index.iterrows()
    ]
    
    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers...")
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {executor.submit(_process_single_subject_mni, task): task[0] for task in tasks}
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)
    
    if all_subject_data:
        return save_study_tensor_mni(all_subject_data, task_name, output_dir, target_sfreq=target_sfreq)
    return None

