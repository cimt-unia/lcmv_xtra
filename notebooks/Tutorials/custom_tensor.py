# Create custom tensor

import lcmv_xtra as lx
from pathlib import Path
import pandas as pd

FS_DIR = Path("/derivatives/_fs")
OUTPUT_DIR = Path("/eeg_tensor")
PROJECT_BASE = Path("path")
CLEAN_DIR = Path("/eeg_clean")

DBS_ROIS = {
    "R1": [10.958, 5.563, -4.948],
    "R2": [28.304, 24.100, -17.535],
    "L1": [-15.983, 6.077, 0.027],
    "L2": [-14.719, -2.180, -0.865],
}

for condition in ["gain", "loss"]:
    df = pd.DataFrame([{
        "subject_id": "DP02",
        "fif_path": str(CLEAN_DIR / f"DP02_{condition}_eeg_raw_eeg.fif")
    }])
    
    lx.assemble_custom_tensor(
        data_index=df,
        fs_dir=FS_DIR,
        output_dir=OUTPUT_DIR,
        roi_coordinates=DBS_ROIS,
        task_name=condition,  
        project_base=PROJECT_BASE,
        mode="single",
        n_jobs=1,
        verbose=True
    )
