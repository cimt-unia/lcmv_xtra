# lcmv_xtra/tensor.py

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from .cimt_atlas import cimt_extraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from .source_estimation import execute_source_estimation

logger = logging.getLogger(__name__)

def make_subject_list(paths: list[str], ids: list[str] = None) -> pd.DataFrame:
    """Create a subject DataFrame from a list of file paths.
    
    Parameters
    ----------
    paths : list of str
        Paths to .fif files.
    ids : list of str, optional
        Subject IDs. Defaults to sub-00, sub-01, ...
    
    Returns
    -------
    pd.DataFrame with columns 'subject_id' and 'fif_path'.
    """
    if ids is None:
        ids = [f"sub-{i:02d}" for i in range(len(paths))]
    return pd.DataFrame({'subject_id': ids, 'fif_path': paths})

def _process_single_subject(args):
    """Helper for parallel processing."""
    sid, fif_path, task_name, project_base, fs_dir, verbose = args
    
    try:
        metadata = execute_source_estimation(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=fif_path,
            fsaverage_dir=fs_dir,
            verbose=verbose
        )
        
        tc, _ = cimt_extraction(
            subject_output_dir=Path(metadata['subject_output']),
            fsaverage_dir=fs_dir,
            verbose=verbose
        )
        
        return {
            "subject_id": sid,
            "time_courses": tc,
            "sfreq": metadata['sfreq_hz'],
            "success": True
        }
    except Exception as e:
        logger.error(f"⚠️ Failed {sid}: {e}")
        return {"subject_id": sid, "success": False}


def scan_eeg_paths(root_dir: Path, pattern: str = "*_c_eeg_mkit_cleaned.fif") -> pd.DataFrame:
    """
    Simple scanner. Finds all files matching the pattern in sub-folders.
    """
    records = []
    root = Path(root_dir)
    
    # Recursively find all files matching the pattern
    for fif_file in root.rglob(pattern):
        # Try to extract subject ID from the parent folder name (sub-XX)
        # or from the filename itself if preferred. Here we assume sub-XX folders.
        subject_folder = fif_file.parent.parent
        if subject_folder.name.startswith("sub-"):
            subject_id = subject_folder.name
        else:
            # Fallback: try to find sub-XX in the path parts
            parts = fif_file.parts
            subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
            
        records.append({
            "subject_id": subject_id,
            "fif_path": str(fif_file.resolve()),
        })
            
    return pd.DataFrame(records)


def save_study_tensor(
    all_subject_data: list, 
    task_name: str, 
    output_dir: Path,
    target_sfreq: float = 250.0 # Default to 250 Hz to preserve data from 250Hz subjects
) -> Path:
    """
    Stacks data into a single 3D array: (Subjects, ROIs, Time).
    Automatically resamples all subjects to 'target_sfreq' for uniformity.
    Uses polyphase filtering (resample_poly) for artifact-free resampling.
    """
    if not all_subject_data: raise ValueError("No data to stack.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resampled_data = []
    
    # 1. Resample each subject to the target frequency
    for d in all_subject_data:
        tc = d['time_courses'] # Shape: (448, T)
        current_sfreq = d['sfreq']
        
        if current_sfreq != target_sfreq:
            # Use resample_poly for better handling of non-periodic biological signals
            # It requires integer up/down sampling factors
            
            # Calculate greatest common divisor to simplify the ratio
            # Note: sfreqs might be floats (e.g. 1000.0), so we cast to int safely
            up = int(target_sfreq)
            down = int(current_sfreq)
            
            # Handle cases where sfreq might not be an integer multiple (rare but possible)
            # If they are not integers, we fall back to calculating duration manually
            if up == target_sfreq and down == current_sfreq:
                gcd = np.gcd(up, down)
                tc_resampled = signal.resample_poly(tc, up // gcd, down // gcd, axis=1)
            else:
                # Fallback for non-integer sfreqs: calculate duration and use resample
                # This is less ideal but handles edge cases like 1024.0 Hz -> 250.0 Hz
                duration_sec = tc.shape[1] / current_sfreq
                n_samples_new = int(np.round(duration_sec * target_sfreq))
                tc_resampled = signal.resample(tc, n_samples_new, axis=1)
                
            resampled_data.append(tc_resampled)
        else:
            resampled_data.append(tc)
            
    # 2. Truncate to shortest time series (in case of slight rounding differences)
    min_t = min([arr.shape[1] for arr in resampled_data])
    
    stacked_data = np.stack([arr[:, :min_t] for arr in resampled_data])
    subject_ids = np.array([d['subject_id'] for d in all_subject_data])
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(output_path, data=stacked_data, subject_ids=subject_ids, sfreq=target_sfreq)
    
    logger.info(f"✅ Saved 3D Tensor {stacked_data.shape} at {target_sfreq} Hz to {output_path}")
    return output_path
    
    
def assemble_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str = "study",
    project_base: Path = None,
    n_jobs: int = -1,
    verbose: bool = False,
    target_sfreq: float = 250.0 # <--- ADDED THIS PARAMETER
) -> Path:
    """
    Takes a list of files, processes them in parallel, and saves ONE .npz file.
    """
    if data_index.empty: return None
    if project_base is None: project_base = Path.cwd()

    tasks = [
        (row['subject_id'], Path(row['fif_path']), task_name, project_base, fs_dir, verbose)
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    if n_jobs == -1: n_jobs = os.cpu_count()
    
    logger.info(f">>> Processing {len(tasks)} subjects with {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {executor.submit(_process_single_subject, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)

    if all_subject_data:
        # <--- PASS target_sfreq TO SAVE FUNCTION
        return save_study_tensor(all_subject_data, task_name, output_dir, target_sfreq=target_sfreq)
    return None

'''
# Usage Examples

import lcmv_xtra as lx
from pathlib import Path

FS_DIR = Path("path/to/fsaverage")
OUTPUT_DIR = Path("path/to/output")
PROJECT_BASE = Path("path/to/project")

# --- Method 1: Auto-scan a directory ---
CLEAN_DIR = Path("path/to/cleaned/files")
df = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_resting_cleaned.fif")
lx.assemble_tensor(df, FS_DIR, OUTPUT_DIR, task_name="resting", project_base=PROJECT_BASE)

# --- Method 2: Manual list of paths ---
df = lx.make_subject_list(
    paths=[
        "/data/sub-01_resting.fif",
        "/data/sub-02_resting.fif",
    ],
    ids=["sub-01", "sub-02"]  # optional, defaults to sub-00, sub-01, ...
)
lx.assemble_tensor(df, FS_DIR, OUTPUT_DIR, task_name="resting", project_base=PROJECT_BASE)

# --- Method 3: Multiple conditions ---
for condition, pattern in [("rest", "*_rest_cleaned.fif"), 
                            ("task", "*_task_cleaned.fif")]:
    df = lx.scan_eeg_paths(CLEAN_DIR, pattern=pattern)
    if not df.empty:
        lx.assemble_tensor(df, FS_DIR, OUTPUT_DIR, 
                          task_name=condition, project_base=PROJECT_BASE)
'''
