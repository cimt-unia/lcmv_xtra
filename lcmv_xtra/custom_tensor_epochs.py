# lcmv_xtra/custom_tensor_epochs.py
"""
Custom MNI ROI Epoch Tensor Assembly.
Extracts time courses from user-defined MNI coordinates per epoch,
producing a 4D tensor: (Subjects, ROIs, Epochs, Time_per_epoch).
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
from nilearn import image

from .source_estimation_epochs import execute_source_estimation_epochs

logger = logging.getLogger(__name__)

METERS_TO_MILLIMETERS: float = 1000.0
DEFAULT_RADIUS_MM: float = 5.0
DEFAULT_TARGET_SFREQ: float = 250.0


def _get_mni_coordinates_from_src(stc: mne.SourceEstimate, src: mne.SourceSpaces) -> np.ndarray:
    """Convert source space vertex coordinates to MNI space."""
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
) -> Tuple[np.ndarray, List[str]]:
    """Extract time courses from user-defined MNI coordinates."""
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


def _process_single_subject_custom_epochs(args: Tuple) -> Dict:
    """Process one subject: epoch source estimation + per-epoch custom ROI extraction."""
    (sid, fif_path, task_name, project_base, fs_dir,
     epoch_duration, roi_coordinates, radius_mm, mode, verbose) = args

    try:
        # 1. Run epoch-based source estimation
        metadata = execute_source_estimation_epochs(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=fif_path,
            fsaverage_dir=fs_dir,
            epoch_duration=epoch_duration,
            verbose=verbose,
        )

        subject_output = Path(metadata['subject_output'])
        n_epochs = metadata['n_epochs']
        src_file = Path(fs_dir) / "fsaverage-vol-5mm-src.fif"
        src = mne.read_source_spaces(str(src_file))

        # 2. Extract custom ROI time courses PER EPOCH
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
        logger.error(f"Failed {sid}: {e}")
        return {"subject_id": sid, "success": False}


def save_custom_tensor_epochs(
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

    output_path = output_dir / f"tensor_{task_name}_custom_epochs.npz"
    np.savez_compressed(
        output_path,
        data=stacked,
        subject_ids=np.array(subject_ids),
        roi_names=np.array(roi_names),
        sfreq=target_sfreq,
        n_epochs=min_epochs,
        n_rois=stacked.shape[1],
    )

    logger.info(
        f"Saved 4D Custom Epoch Tensor {stacked.shape} "
        f"(Subjects, ROIs, Epochs, Time) at {target_sfreq} Hz to {output_path}"
    )
    return output_path


def assemble_custom_tensor_epochs(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    roi_coordinates: Dict[str, List[float]],
    task_name: str = "study",
    project_base: Optional[Path] = None,
    epoch_duration: float = 2.0,
    radius_mm: float = DEFAULT_RADIUS_MM,
    mode: Literal["sphere", "single"] = "sphere",
    n_jobs: int = -1,
    verbose: bool = False,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> Optional[Path]:
    """
    Assemble a 4D custom ROI epoch tensor from cleaned continuous EEG files.

    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'fif_path' columns.
    fs_dir : Path
        Path to fsaverage directory.
    output_dir : Path
        Where to save the final .npz tensor.
    roi_coordinates : dict
        Mapping of ROI name → [x, y, z] MNI coordinates.
    task_name : str
        Label for the output file.
    project_base : Path, optional
        Base project directory. Defaults to cwd.
    epoch_duration : float
        Length of each non-overlapping epoch in seconds.
    radius_mm : float
        Sphere radius for voxel averaging around each coordinate.
    mode : 'sphere' or 'single'
        'sphere' averages all voxels within radius_mm.
        'single' uses only the closest voxel.
    n_jobs : int
        Number of parallel workers. -1 uses all CPUs.
    verbose : bool
        Enable verbose logging.
    target_sfreq : float
        Target sampling rate for the output tensor.

    Returns
    -------
    Path to saved .npz file, or None if no subjects succeeded.
    """
    if data_index.empty:
        return None
    if project_base is None:
        project_base = Path.cwd()

    tasks = [
        (
            row['subject_id'], Path(row['fif_path']), task_name,
            project_base, fs_dir, epoch_duration,
            roi_coordinates, radius_mm, mode, verbose,
        )
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject_custom_epochs, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)

    if all_subject_data:
        return save_custom_tensor_epochs(
            all_subject_data, task_name, output_dir, target_sfreq=target_sfreq,
        )
    return None

'''
import lcmv_xtra as lx
from pathlib import Path

FS_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/_fs")
OUTPUT_DIR = Path("/mnt/movement/users/jaizor/xtra/notebooks/EEG/UNI/Epochs/data")
PROJECT_BASE = Path("/mnt/movement/users/jaizor/xtra/derivatives/eeg/uni")
EPOCHS_DIR = PROJECT_BASE / "eeg_epochs"

# Define your custom ROIs as MNI coordinates
ROI_COORDINATES = {
    "STN_L": [-11.89, -14.51, -6.40],
    "STN_R": [12.53, -13.97, -6.57],
    "M1_L":  [-38, -24, 58],
    "M1_R":  [38, -24, 58],
    "SMA":   [0, -10, 65],
}

df = lx.scan_eeg_paths(EPOCHS_DIR, pattern="*_pre_raw.fif")
df = df[~df['subject_id'].str.contains("sub-01")]
df['subject_id'] = df['fif_path'].apply(
    lambda p: Path(p).stem.split('_sm_eeg')[0]
)

lx.assemble_custom_tensor_epochs(
    data_index=df,
    fs_dir=FS_DIR,
    output_dir=OUTPUT_DIR,
    roi_coordinates=ROI_COORDINATES,
    task_name="sm_pre",
    project_base=PROJECT_BASE,
    epoch_duration=2.5,
    radius_mm=5.0,
    mode="sphere",
    target_sfreq=250.0,
    n_jobs=-1,
    verbose=True,
)
'''
