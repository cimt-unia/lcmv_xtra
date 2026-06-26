# lcmv_stats/tensor.py
"""
Tools for discovering FIF files and aggregating source-space data 
into ML-ready 3D tensors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import lcmv_xtra components directly into the library
from lcmv_xtra import execute_source_estimation
from lcmv_xtra.cimt_atlas import cimt_extraction

logger = logging.getLogger(__name__)


def build_fif_manifest(root_dir: Path, task_name: str = "rest_off") -> pd.DataFrame:
    """
    Scans a complex directory structure and returns a clean DataFrame of FIF paths.
    """
    records = []
    root = Path(root_dir)
    
    # Look for sub-XX folders
    for sub_folder in root.glob("sub-*"):
        if not sub_folder.is_dir():
            continue
            
        subject_id = sub_folder.name 
        
        # Look for the task folder inside
        task_folder = sub_folder / task_name
        if not task_folder.exists(): 
            continue
        
        # Find all _c_ (central/cleaned) FIF files
        for fif_file in task_folder.glob(f"*{task_name}*_*_c_eeg_mkit_cleaned.fif"):
            records.append({
                "subject_id": subject_id,
                "task_name": task_name,
                "fif_path": str(fif_file.resolve()),
                "condition_type": "rest" 
            })
            
    if not records:
        logger.warning(f"No FIF files found for task '{task_name}' in {root_dir}")
        
    return pd.DataFrame(records)


def save_study_tensor(all_subject_data: list, task_name: str, output_dir: Path) -> Path:
    """
    Stacks a list of subject dictionaries into a 3D tensor and saves as .npz.
    """
    if not all_subject_data:
        raise ValueError("No subject data provided to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle varying time dimensions by truncating to the shortest subject
    min_t = min([d['time_courses'].shape[1] for d in all_subject_data])
    logger.info(f"Truncating all subjects to {min_t} time samples for uniform tensor shape.")
    
    stacked_data = np.stack([d['time_courses'][:, :min_t] for d in all_subject_data])
    subject_ids = np.array([d['subject_id'] for d in all_subject_data])
    sfreq = all_subject_data[0]['sfreq'] # Assuming uniform sampling rate
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(
        output_path, 
        data=stacked_data, 
        subject_ids=subject_ids,
        sfreq=sfreq
    )
    
    logger.info(f"✅ Saved Master Tensor: {stacked_data.shape} to {output_path}")
    return output_path


def build_study_tensors(
    fif_manifest: pd.DataFrame,
    project_base: Path,
    fs_dir: Path,
    output_dir: Path,
    task_name: str,
    verbose: bool = False
) -> Path:
    """
    Master orchestrator: Processes FIF files from a manifest using lcmv_xtra,
    extracts CIMT time courses, and aggregates them into a single 3D .npz tensor.
    
    Args:
        fif_manifest: DataFrame from build_fif_manifest.
        project_base: Root directory for the project (used for derivative paths).
        fs_dir: Path to fsaverage resources.
        output_dir: Directory to save the final .npz tensor.
        task_name: Name of the task/condition.
        verbose: Enable detailed logging from lcmv_xtra.
        
    Returns:
        Path to the saved .npz file, or None if no subjects were processed.
    """
    if fif_manifest.empty:
        logger.error("The provided manifest is empty. Nothing to process.")
        return None

    all_subject_data = []
    
    for _, row in fif_manifest.iterrows():
        sid = row['subject_id']
        fif_path = Path(row['fif_path'])
        
        logger.info(f">>> Processing {sid} from {fif_path.name}...")
        
        try:
            # 1. Run Source Estimation (lcmv_xtra handles the absolute path correctly)
            metadata = execute_source_estimation(
                project_base=project_base,
                subject_id=sid,
                task=task_name,
                ica_file_path=fif_path, 
                fsaverage_dir=fs_dir,
                verbose=verbose
            )
            
            # 2. Extract CIMT Time Courses (lcmv_xtra)
            tc, _ = cimt_extraction(
                subject_output_dir=Path(metadata['subject_output']),
                fsaverage_dir=fs_dir,
                verbose=verbose
            )
            
            # 3. Store in memory for tensor stacking
            all_subject_data.append({
                "subject_id": sid,
                "time_courses": tc,
                "sfreq": metadata['sfreq_hz']
            })
            
        except Exception as e:
            logger.error(f"⚠️ Failed {sid}: {e}")
            continue

    # 4. Build and save the tensor
    if all_subject_data:
        logger.info(f">>> Aggregating {len(all_subject_data)} successful subjects...")
        return save_study_tensor(all_subject_data, task_name, output_dir)
    else:
        logger.error("❌ No subjects were successfully processed.")
        return None
