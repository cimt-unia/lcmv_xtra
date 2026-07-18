# =============================================================================
# Coregistration Check Tutorial
# =============================================================================
"""
After running LCMV source estimation, use this to check that your EEG
electrodes are correctly aligned to the head model.

WHAT YOU'LL SEE:
  • A 3D head you can rotate/zoom
  • Red dots = your EEG electrodes
  • Green dot = Nasion (bridge of nose)
  • Blue dot = LPA (by left ear)
  • Red dot = RPA (by right ear)
  • The title shows coregistration error (lower = better, <5mm is good)

WHAT YOU NEED — 4 files:
  1. Your cleaned EEG .fif file
  2. fsaverage-trans.fif         (created by the LCMV pipeline)
  3. fsaverage-head-dense.fif    (from your fsaverage folder)
  4. pipeline_metadata.json      (created by the LCMV pipeline)

WHERE TO FIND THEM:
  Your fsaverage folder (let's call it _fs):
    _fs/fsaverage/
      ├── bem/fsaverage-head-dense.fif        ← file #3
      └── surf/
          ├── lh.pial                          ← optional, for brain view
          └── rh.pial                          ← optional, for brain view

  Your LCMV output (one per subject):
    derivatives/lcmv/<subject>_<task>/
      ├── fsaverage-trans.fif                  ← file #2
      └── pipeline_metadata.json               ← file #4

  Your cleaned data:
    derivatives/eeg/clean/<subject>_cleaned.fif ← file #1
"""

from lcmv_xtra import plot_coregistration
from pathlib import Path

# =============================================================================
# STEP 1: Set your base paths (change these)
# =============================================================================
FS_DIR = Path("/derivatives/_fs/fsaverage")
LCMV_DIR = Path("/lcmv/Sub08_ocd_neutral")
FIF_PATH = Path("/derivatives/ocd/trials/sub08/clean/Sub08_Neutral_proc.fif")

# =============================================================================
# STEP 2: Basic check — just head + electrodes
# =============================================================================
plot_coregistration(
    fif_path=FIF_PATH,
    trans_file=LCMV_DIR / "fsaverage-trans.fif",
    head_surf_file=FS_DIR / "bem" / "fsaverage-head-dense.fif",
    pipeline_metadata_file=LCMV_DIR / "pipeline_metadata.json",
)

# =============================================================================
# STEP 3: Add the brain inside the head
# =============================================================================
plot_coregistration(
    fif_path=FIF_PATH,
    trans_file=LCMV_DIR / "fsaverage-trans.fif",
    head_surf_file=FS_DIR / "bem" / "fsaverage-head-dense.fif",
    pipeline_metadata_file=LCMV_DIR / "pipeline_metadata.json",
    add_brain=True,
    lh_pial_file=FS_DIR / "surf" / "lh.pial",
    rh_pial_file=FS_DIR / "surf" / "rh.pial",
)

