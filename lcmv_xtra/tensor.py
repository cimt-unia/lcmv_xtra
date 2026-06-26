# lcmv_xtra/tensor.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import lcmv_xtra
from .source_estimation import execute_source_estimation
from .cimt_atlas import cimt_extraction

logger = logging.getLogger(__name__)


def scan_eeg_paths(root_dir: Path, task_name: str = "rest_off") -> pd.DataFrame:
    """
    Recursively scans a directory for cleaned FIF files and returns 
    a structured DataFrame of paths and metadata.
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
                "fif_path": str(fif_file.resolve()), # Absolute path
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
    sfreq = all_subject_data[0]['sfreq'] 
    
    output_path = output_dir / f"study_{task_name}.npz"
    np.savez_compressed(
        output_path, 
        data=stacked_data, 
        subject_ids=subject_ids,
        sfreq=sfreq
    )
    
    logger.info(f"✅ Saved Master Tensor: {stacked_data.shape} to {output_path}")
    return output_path


def assemble_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str,
    derivatives_root: Path = None, # Optional: Where to save intermediate LCMV files
    verbose: bool = False
) -> Path:
    """
    Master orchestrator: Processes FIF files from a data index using lcmv_xtra,
    extracts CIMT time courses, and aggregates them into a single 3D .npz tensor.
    
    Args:
        data_index: DataFrame from scan_eeg_paths.
        fs_dir: Path to fsaverage resources (must contain fsaverage-vol-5mm-src.fif).
        output_dir: Directory to save the final .npz tensor.
        task_name: Name of the task/condition.
        derivatives_root: (Optional) Root folder for intermediate LCMV derivatives. 
                          If None, uses the current working directory.
        verbose: Enable detailed logging.
        
    Returns:
        Path to the saved .npz file.
    """
    if data_index.empty:
        logger.error("The provided data index is empty. Nothing to process.")
        return None

    if derivatives_root is None:
        derivatives_root = Path.cwd()

    all_subject_data = []
    
    for _, row in data_index.iterrows():
        sid = row['subject_id']
        fif_path = Path(row['fif_path'])
        
        logger.info(f">>> Processing {sid} from {fif_path.name}...")
        
        try:
            # 1. Run Source Estimation
            # We pass the absolute fif_path directly. 
            # We use a generic project_base just to satisfy the function signature, 
            # but the actual input file is the absolute one.
            metadata = execute_source_estimation(
                project_base=derivatives_root, 
                subject_id=sid,
                task=task_name,
                ica_file_path=fif_path, # Absolute path overrides project_base logic
                fsaverage_dir=fs_dir,
                verbose=verbose
            )
            
            # 2. Extract CIMT Time Courses
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

'''
# Usage Example
import lcmv_xtra as lx
from pathlib import Path

# 1. Define your paths
CLEAN_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/eeg/rest/clean")
FS_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/_fs")
OUTPUT_DIR = Path("./ml_data")

# 2. Scan for files (This gives you absolute paths)
df_index = lx.scan_eeg_paths(CLEAN_DIR, "rest_off")

# 3. Process and build tensor
lx.assemble_tensor(
    data_index=df_index,
    fs_dir=FS_DIR,          # Only FS dir is strictly required for resources
    output_dir=OUTPUT_DIR,  # Where the .npz goes
    task_name="rest_off",
    verbose=False
)
'''
