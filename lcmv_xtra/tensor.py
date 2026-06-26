# lcmv_xtra/tensor.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from .source_estimation import execute_source_estimation
from .cimt_atlas import cimt_extraction

logger = logging.getLogger(__name__)

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


def save_study_tensor(all_subject_data: list, task_name: str, output_dir: Path) -> Path:
    """Stacks data into a single 3D array: (Subjects, ROIs, Time)."""
    if not all_subject_data: raise ValueError("No data to stack.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Truncate to shortest time series so they all fit in one array
    min_t = min([d['time_courses'].shape[1] for d in all_subject_data])
    
    stacked_data = np.stack([d['time_courses'][:, :min_t] for d in all_subject_data])
    subject_ids = np.array([d['subject_id'] for d in all_subject_data])
    sfreq = all_subject_data[0]['sfreq']
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(output_path, data=stacked_data, subject_ids=subject_ids, sfreq=sfreq)
    
    logger.info(f"✅ Saved 3D Tensor {stacked_data.shape} to {output_path}")
    return output_path


def assemble_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str = "study",
    project_base: Path = None,
    n_jobs: int = -1,
    verbose: bool = False
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
        return save_study_tensor(all_subject_data, task_name, output_dir)
    return None

'''
# Usage Example


import lcmv_xtra as lx
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CLEAN_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/eeg/rest/clean")
FS_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/_fs")
OUTPUT_DIR = Path("./ml_data")
PROJECT_BASE = Path("/mnt/movement/users/jaizor/xtra")

# =============================================================================
# 2. BUILD TENSORS (One per condition)
# =============================================================================

# --- LEFT HAND ---
print(">>> Processing Left Hand...")
df_left = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_l_eeg_mkit_cleaned.fif")

if not df_left.empty:
    lx.assemble_tensor(
        data_index=df_left,
        fs_dir=FS_DIR,
        output_dir=OUTPUT_DIR,
        task_name="left_hand", # Saves as study_left_hand.npz
        project_base=PROJECT_BASE,
        n_jobs=-1
    )
else:
    print("No Left Hand files found.")

# --- RIGHT HAND ---
print("\n>>> Processing Right Hand...")
df_right = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_r_eeg_mkit_cleaned.fif")

if not df_right.empty:
    lx.assemble_tensor(
        data_index=df_right,
        fs_dir=FS_DIR,
        output_dir=OUTPUT_DIR,
        task_name="right_hand", # Saves as study_right_hand.npz
        project_base=PROJECT_BASE,
        n_jobs=-1
    )
else:
    print("No Right Hand files found.")

# --- EYES CLOSED (Central) ---
print("\n>>> Processing Eyes Closed (Central)...")
df_closed = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_c_eeg_mkit_cleaned.fif")

if not df_closed.empty:
    lx.assemble_tensor(
        data_index=df_closed,
        fs_dir=FS_DIR,
        output_dir=OUTPUT_DIR,
        task_name="eyes_closed", # Saves as study_eyes_closed.npz
        project_base=PROJECT_BASE,
        n_jobs=-1
    )
else:
    print("No Central/Eyes Closed files found.")

print("\n✅ All tensors saved to ./ml_data/")
'''
