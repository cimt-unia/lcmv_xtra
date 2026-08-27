# lcmv_xtra/tensor_epochs.py
# (Subjects, ROIs, Epochs, Time)

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .cimt_atlas import cimt_extraction
from .source_estimation_epochs import execute_source_estimation_epochs

logger = logging.getLogger(__name__)


def make_subject_list(paths: List[str], ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Create a subject DataFrame from a list of file paths."""
    if ids is None:
        ids = [f"sub-{i:02d}" for i in range(len(paths))]
    return pd.DataFrame({'subject_id': ids, 'fif_path': paths})


def scan_eeg_paths(root_dir: Path, pattern: str = "*_c_eeg_mkit_cleaned.fif") -> pd.DataFrame:
    """Find all files matching the pattern in sub-folders."""
    records = []
    root = Path(root_dir)
    
    for fif_file in root.rglob(pattern):
        subject_folder = fif_file.parent.parent
        if subject_folder.name.startswith("sub-"):
            subject_id = subject_folder.name
        else:
            parts = fif_file.parts
            subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
            
        records.append({
            "subject_id": subject_id,
            "fif_path": str(fif_file.resolve()),
        })
            
    return pd.DataFrame(records)


def _process_single_subject_epochs(args: Tuple) -> Dict:
    """Process one subject: epoch source estimation + per-epoch atlas extraction."""
    (
        sid, fif_path, task_name, project_base, fs_dir,
        epoch_duration, verbose,
        noise_cov_method, baseline_tmin, baseline_tmax
    ) = args
    
    try:
        # 1. Run epoch-based source estimation with proper noise covariance
        metadata = execute_source_estimation_epochs(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=fif_path,
            fsaverage_dir=fs_dir,
            epoch_duration=epoch_duration,
            verbose=verbose,
            noise_cov_method=noise_cov_method,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax
        )
        
        subject_output = Path(metadata['subject_output'])
        n_epochs = metadata['n_epochs']
        
        # 2. Extract time courses PER EPOCH
        epoch_time_courses = []
        for i in range(n_epochs):
            stc_file = subject_output / f'source_estimate_LCMV_epoch_{i:03d}.h5'
            if not stc_file.exists():
                logger.warning(f"Missing epoch STC for {sid} epoch {i}, skipping")
                continue
                
            tc, _ = cimt_extraction(
                subject_output_dir=subject_output,
                fsaverage_dir=fs_dir,
                stc_filename=f'source_estimate_LCMV_epoch_{i:03d}.h5',
                verbose=False
            )
            epoch_time_courses.append(tc)  # Shape: (n_rois, time_per_epoch)
        
        if not epoch_time_courses:
            raise RuntimeError(f"No valid epoch STCs found for {sid}")
            
        return {
            "subject_id": sid,
            "epoch_time_courses": np.stack(epoch_time_courses, axis=0),
            "sfreq": metadata['sfreq_hz'],
            "n_epochs": len(epoch_time_courses),
            "success": True
        }
    except Exception as e:
        logger.error(f"Failed {sid}: {e}")
        return {"subject_id": sid, "success": False}


def save_study_tensor_epochs(
    all_subject_data: List[Dict], 
    task_name: str, 
    output_dir: Path,
    target_sfreq: float = 250.0
) -> Path:
    """
    Stack epoch data into a 4D array: (Subjects, ROIs, Epochs, Time_per_epoch).
    Resamples to target_sfreq and truncates to shortest common dimensions.
    """
    if not all_subject_data:
        raise ValueError("No data to stack.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resampled_data = []
    subject_ids = []
    
    # 1. Resample each subject's epochs to target frequency
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
    
    # 2. Truncate to shortest common epoch count and time length
    min_epochs = min(arr.shape[0] for arr in resampled_data)
    min_time = min(arr.shape[2] for arr in resampled_data)
    
    stacked = np.stack([
        arr[:min_epochs, :, :min_time] for arr in resampled_data
    ])  # Shape: (n_subjects, min_epochs, n_rois, min_time)
    
    # Reorder to (Subjects, ROIs, Epochs, Time) for consistency with continuous tensor
    stacked = np.transpose(stacked, (0, 2, 1, 3))
    
    subject_ids_arr = np.array(subject_ids)
    
    output_path = output_dir / f"study_{task_name}_epochs.npz"
    np.savez_compressed(
        output_path, 
        data=stacked, 
        subject_ids=subject_ids_arr, 
        sfreq=target_sfreq,
        epoch_duration=all_subject_data[0].get('epoch_duration', None),
        n_epochs=min_epochs,
        n_rois=stacked.shape[1]
    )
    
    logger.info(
        f"Saved 4D Epoch Tensor {stacked.shape} "
        f"(Subjects, ROIs, Epochs, Time) at {target_sfreq} Hz to {output_path}"
    )
    return output_path


def assemble_tensor_epochs(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str = "study",
    project_base: Optional[Path] = None,
    epoch_duration: float = 2.0,
    n_jobs: int = -1,
    verbose: bool = False,
    target_sfreq: float = 250.0,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: float = 0.1
) -> Optional[Path]:
    """
    Assemble a 4D epoch tensor from cleaned continuous EEG files.
    
    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'fif_path' columns.
    fs_dir : Path
        Path to fsaverage directory.
    output_dir : Path
        Where to save the final .npz tensor.
    task_name : str
        Label for the output file.
    project_base : Path, optional
        Base project directory. Defaults to cwd.
    epoch_duration : float
        Length of each non-overlapping epoch in seconds. Default 2.0.
    n_jobs : int
        Number of parallel workers. -1 uses all CPUs.
    verbose : bool
        Enable verbose logging.
    target_sfreq : float
        Target sampling rate for the output tensor.
    noise_cov_method : str
        Estimator for noise covariance ('shrunk', 'oas', 'empirical').
        Passed to execute_source_estimation_epochs. Default 'shrunk'.
    baseline_tmin : float | None
        Start of baseline window for noise cov (seconds relative to epoch onset).
        None defaults to 0.0. Passed to execute_source_estimation_epochs.
    baseline_tmax : float
        End of baseline window for noise cov (seconds relative to epoch onset).
        Must be < epoch_duration. Default 0.1.
        
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
            row['subject_id'], 
            Path(row['fif_path']), 
            task_name, 
            project_base, 
            fs_dir, 
            epoch_duration,
            verbose,
            noise_cov_method,
            baseline_tmin,
            baseline_tmax
        )
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    logger.info(f"Processing {len(tasks)} subjects with {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject_epochs, task): task[0] 
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)

    if all_subject_data:
        return save_study_tensor_epochs(
            all_subject_data, task_name, output_dir, target_sfreq=target_sfreq
        )
    return None


'''
import lcmv_xtra as lx
from pathlib import Path

FS_DIR = Path("/path/to/fsaverage")
OUTPUT_DIR = Path("/path/to/output")
PROJECT_BASE = Path("/path/to/project")
CLEAN_DIR = Path("/path/to/cleaned/files")

# Scan and assemble with proper noise covariance separation
df = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_resting_cleaned.fif")
lx.assemble_tensor_epochs(
    data_index=df,
    fs_dir=FS_DIR,
    output_dir=OUTPUT_DIR,
    task_name="resting",
    project_base=PROJECT_BASE,
    epoch_duration=2.0,
    target_sfreq=250.0,
    n_jobs=-1,
    verbose=True,
    noise_cov_method='shrunk',     # Ledoit-Wolf for short baselines
    baseline_tmin=None,            # Baseline starts at epoch onset
    baseline_tmax=0.1              # 100ms baseline for noise estimation
)
'''
