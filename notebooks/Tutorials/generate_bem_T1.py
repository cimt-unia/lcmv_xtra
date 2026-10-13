"""
Native-Space Anatomical Pre-processing Pipeline
Generates BEM and 5mm brain-bounded volume source space using the subject's own T1.
Includes robust fallback mechanisms for watershed BEM failures.
"""
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import List

import numpy as np
import mne

# ============================================================================
# ENVIRONMENT SETUP (Must precede MNE import)
# ============================================================================
FS_HOME = '/mnt/movement/users/jaizor/xtra/software/freesurfer'
SUBJECTS_DIR_ENV = '/mnt/movement/users/jaizor/xtra/derivatives/_fs'

os.environ['FREESURFER_HOME'] = FS_HOME
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR_ENV

# Add FreeSurfer bin to PATH so mri_watershed and other binaries are found
fs_bin = os.path.join(FS_HOME, 'bin')
current_path = os.environ.get('PATH', '')
if fs_bin not in current_path:
    os.environ['PATH'] = fs_bin + os.pathsep + current_path

# ============================================================================
# CONFIGURATION
# ============================================================================
T1_FILE = Path("preproc_T1w.nii.gz")
SUBJECT_ID = "sub-01"
SUBJECTS_DIR = Path(SUBJECTS_DIR_ENV)
FREESURFER_HOME = Path(FS_HOME)

# Explicit path to conda env bin to ensure compatible tcsh is found in Jupyter
XTRA_BIN_PATH = "/mnt/movement/users/jaizor/xtra/miniconda3/envs/Xtra/bin"

BEM_CONDUCTIVITY: List[float] = [0.3, 0.006, 0.3]  # Brain, Skull, Scalp (S/m)
BEM_ICO: int = 4                                   # ~5120 vertices per surface
SOURCE_SPACE_POS: float = 5.0                      # 5mm grid spacing
MINDIST: float = 5.0                               # Min distance from inner skull (mm)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Watershed fallback configurations
WATERSHED_ATTEMPTS = [
    dict(atlas=False, gcaatlas=False, preflood=None),  # MNE default (25)
    dict(atlas=False, gcaatlas=False, preflood=15),
    dict(atlas=False, gcaatlas=False, preflood=10),
    dict(atlas=True,  gcaatlas=False, preflood=None),  # atlas without GCA
]

# ============================================================================
# PHASE 1: FREESURFER RECON-ALL
# ============================================================================
def run_recon_all(t1_path: Path, subject: str, subjects_dir: Path, fs_home: Path) -> None:
    """Execute FreeSurfer recon-all with real-time output streaming."""
    if not t1_path.exists():
        raise FileNotFoundError(f"T1 file not found: {t1_path}")

    subj_dir = subjects_dir / subject
    aparc_aseg = subj_dir / "mri" / "aparc+aseg.mgz"

    if aparc_aseg.exists():
        logger.info(f"recon-all already completed for {subject}. Skipping.")
        return

    logger.info(f"Starting recon-all for {subject}...")

    # Prepend Xtra bin path so subprocess finds the symlinked tcsh
    bash_cmd = f"""
    export PATH={XTRA_BIN_PATH}:$PATH
    export FREESURFER_HOME={fs_home}
    source {fs_home}/SetUpFreeSurfer.sh
    export SUBJECTS_DIR={subjects_dir}
    recon-all -i {t1_path} -s {subject} -all
    """

    process = subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            logger.info(line.strip())

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"recon-all failed with exit code {process.returncode}")

    logger.info("recon-all completed successfully.")

# ============================================================================
# PHASE 2: ROBUST BEM & SOURCE SPACE GENERATION
# ============================================================================
def _try_bem_model(subject: str, subjects_dir: Path):
    """Return a 3-layer BEM model at ico=4, or None if surfaces intersect."""
    try:
        return mne.make_bem_model(
            subject=subject, ico=BEM_ICO,
            conductivity=BEM_CONDUCTIVITY, subjects_dir=str(subjects_dir),
        )
    except RuntimeError as e:
        if "not completely inside" in str(e):
            logger.warning(f"Surface check failed: {e}")
            return None
        raise

def _rebuild_outer_skull(subject: str, subjects_dir: Path, thickness_mm: float = 5.0):
    """Fallback: outer skull = inner skull pushed outward along normals."""
    bem_dir = subjects_dir / subject / "bem"
    
    # Locate the inner skull surface (copied to bem/ or left in watershed/)
    inner_skull_path = bem_dir / "inner_skull_surface"
    if not inner_skull_path.exists():
        inner_skull_path = bem_dir / "watershed" / f"{subject}_inner_skull_surface"
        
    surf_dict = mne.read_surface(str(inner_skull_path), return_dict=True)[2]
    surf_dict = mne.surface.complete_surface_info(surf_dict, copy=True, verbose=False)
    
    # CRITICAL: MNE surface coordinates are in METERS. Convert mm to m.
    thickness_m = thickness_mm / 1000.0
    new_rr = surf_dict["rr"] + thickness_m * surf_dict["nn"]
    
    out_path = bem_dir / "outer_skull_surface"
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
        
    mne.write_surface(str(out_path), new_rr, surf_dict["tris"], overwrite=True)
    logger.warning(f"Rebuilt outer_skull_surface as inner skull + {thickness_mm} mm (normals offset).")

def generate_bem_and_source_space(subject: str, subjects_dir: Path) -> None:
    """Generate BEM solution and bounded 5mm volume source space with robust fallbacks."""
    subj_dir = subjects_dir / subject
    bem_dir = subj_dir / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)

    bem_sol_file = bem_dir / f"{subject}-5120-5120-5120-bem-sol.fif"
    src_file = subj_dir / f"{subject}-vol-5mm-src.fif"

    bem_model = None
    for params in WATERSHED_ATTEMPTS:
        logger.info(f"Watershed attempt: {params}")
        mne.bem.make_watershed_bem(
            subject=subject, subjects_dir=str(subjects_dir),
            overwrite=True, copy=True, **params,
        )
        bem_model = _try_bem_model(subject, subjects_dir)
        if bem_model is not None:
            logger.info("BEM topology check passed.")
            break

    if bem_model is None:
        logger.warning("All watershed attempts failed; offsetting inner skull to create outer skull.")
        _rebuild_outer_skull(subject, subjects_dir, thickness_mm=5.0)
        bem_model = _try_bem_model(subject, subjects_dir)

    if bem_model is None:
        raise RuntimeError(
            "Could not build a valid subject BEM. Inspect T1.mgz / "
            "talairach_with_skull.lta or the generated surfaces."
        )

    logger.info("Computing BEM solution...")
    bem_sol = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(str(bem_sol_file), bem_sol, overwrite=True)

    logger.info(f"Creating {SOURCE_SPACE_POS}mm volume source space (brain-bounded)...")
    src = mne.setup_volume_source_space(
        subject=subject, subjects_dir=str(subjects_dir),
        pos=SOURCE_SPACE_POS, bem=bem_sol, mindist=MINDIST,
        add_interpolator=True,
    )
    mne.write_source_spaces(str(src_file), src, overwrite=True)
    
    n_sources = sum(s['nuse'] for s in src)
    logger.info(f"Anatomy preparation complete: {n_sources} sources at {SOURCE_SPACE_POS}mm.")

    # Visual QA: always check this before trusting the BEM
    try:
        logger.info("Generating BEM visual QA...")
        fig = mne.viz.plot_bem(
            subject=subject, subjects_dir=str(subjects_dir),
            brain_surfaces="white", orientation="coronal", show=False
        )
        qa_path = subj_dir / f"{subject}_bem_check.png"
        fig.savefig(qa_path, dpi=150)
        logger.info(f"BEM QA plot saved to {qa_path}")
    except Exception as e:
        logger.warning(f"Could not generate BEM QA plot: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main() -> None:
    """Main execution pipeline."""
    if not FREESURFER_HOME.joinpath("SetUpFreeSurfer.sh").exists():
        raise FileNotFoundError(f"FreeSurfer not found at {FREESURFER_HOME}")

    logger.info(f"FreeSurfer verified at {FREESURFER_HOME}")
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: FreeSurfer cortical reconstruction (~3-5 hours)
    run_recon_all(T1_FILE, SUBJECT_ID, SUBJECTS_DIR, FREESURFER_HOME)

    # Phase 2: MNE-Python BEM and source space generation (~2-5 minutes)
    generate_bem_and_source_space(SUBJECT_ID, SUBJECTS_DIR)

    logger.info("=" * 60)
    logger.info("Pipeline finished successfully.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
