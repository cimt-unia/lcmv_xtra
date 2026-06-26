# lcmv_xtra/tensor.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from .source_estimation import execute_source_estimation
from .cimt_atlas import cimt_extraction

logger = logging.getLogger(__name__)

def _process_single_subject(args):
    """
    Helper function to be run in parallel. 
    Takes a tuple of arguments to avoid pickle issues with complex objects.
    """
    sid, fif_path, task_name, project_base, fs_dir, verbose = args
    
    try:
        # 1. Run Source Estimation
        metadata = execute_source_estimation(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=fif_path,
            fsaverage_dir=fs_dir,
            verbose=verbose
        )
        
        # 2. Extract CIMT Time Courses
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


def scan_eeg_paths(root_dir: Path, task_name: str = "rest_off") -> pd.DataFrame:
    records = []
    root = Path(root_dir)
    for sub_folder in root.glob("sub-*"):
        if not sub_folder.is_dir(): continue
        subject_id = sub_folder.name 
        task_folder = sub_folder / task_name
        if not task_folder.exists(): continue
        for fif_file in task_folder.glob(f"*{task_name}*_*_c_eeg_mkit_cleaned.fif"):
            records.append({
                "subject_id": subject_id,
                "task_name": task_name,
                "fif_path": str(fif_file.resolve()),
            })
    return pd.DataFrame(records)


def save_study_tensor(all_subject_data: list, task_name: str, output_dir: Path) -> Path:
    if not all_subject_data: raise ValueError("No data to stack.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_t = min([d['time_courses'].shape[1] for d in all_subject_data])
    stacked_data = np.stack([d['time_courses'][:, :min_t] for d in all_subject_data])
    subject_ids = np.array([d['subject_id'] for d in all_subject_data])
    sfreq = all_subject_data[0]['sfreq']
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(output_path, data=stacked_data, subject_ids=subject_ids, sfreq=sfreq)
    logger.info(f"✅ Saved Master Tensor: {stacked_data.shape} to {output_path}")
    return output_path


def assemble_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str,
    project_base: Path = None,
    n_jobs: int = -1, # Use all available cores by default
    verbose: bool = False
) -> Path:
    """
    Parallelized orchestrator for building study tensors.
    """
    if data_index.empty: return None
    if project_base is None: project_base = Path.cwd()

    # Prepare arguments for parallel execution
    tasks = [
        (row['subject_id'], Path(row['fif_path']), task_name, project_base, fs_dir, verbose)
        for _, row in data_index.iterrows()
    ]

    all_subject_data = []
    
    # Determine number of workers
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    
    logger.info(f">>> Starting parallel processing with {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {executor.submit(_process_single_subject, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)
            else:
                logger.warning(f"Subject {result['subject_id']} failed and was skipped.")

    if all_subject_data:
        return save_study_tensor(all_subject_data, task_name, output_dir)
    return None

'''
import lcmv_xtra as lx
from pathlib import Path

CLEAN_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/eeg/rest/clean")
FS_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/_fs")
OUTPUT_DIR = Path("./ml_data")
PROJECT_BASE = Path("/mnt/movement/users/jaizor/xtra")

df_index = lx.scan_eeg_paths(CLEAN_DIR, "rest_off")

# It runs in parallel automatically
lx.assemble_tensor(
    data_index=df_index,
    fs_dir=FS_DIR,
    output_dir=OUTPUT_DIR,
    task_name="rest_off",
    project_base=PROJECT_BASE,
    n_jobs=-1 # Use all cores
)
'''


