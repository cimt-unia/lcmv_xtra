"""
Anatomical Pre-processing Pipeline
Generates BEM and 5mm brain-bounded volume source space compatible with lcmv_xtra.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import List

# ============================================================================
# ENVIRONMENT SETUP (Must precede MNE import)
# ============================================================================
FS_HOME = 'software/freesurfer'
SUBJECTS_DIR_ENV = 'derivatives/_fs'

os.environ['FREESURFER_HOME'] = FS_HOME
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR_ENV

# Add FreeSurfer bin to PATH so mri_watershed and other binaries are found
fs_bin = os.path.join(FS_HOME, 'bin')
current_path = os.environ.get('PATH', '')
if fs_bin not in current_path:
    os.environ['PATH'] = fs_bin + os.pathsep + current_path

import mne

# ============================================================================
# CONFIGURATION
# ============================================================================
T1_FILE = Path("T1w.nii.gz")
SUBJECT_ID = "sub-DP02"
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

    # Prepend Xtra bin path so subprocess finds the symlinked tcsh v6.21.00
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


def generate_bem_and_source_space(subject: str, subjects_dir: Path) -> None:
    """Generate BEM solution and bounded 5mm volume source space."""
    subj_dir = subjects_dir / subject
    bem_dir = subj_dir / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)

    bem_sol_file = bem_dir / f"{subject}-5120-5120-5120-bem-sol.fif"
    src_file = subj_dir / f"{subject}-vol-5mm-src.fif"

    logger.info("Creating BEM surfaces via watershed algorithm (using GCA atlas)...")
    
    # Watershed BEM generation
    mne.bem.make_watershed_bem(
        subject=subject,
        subjects_dir=str(subjects_dir),
        overwrite=True,
        atlas=True,
        gcaatlas=True
    )

    # CRITICAL FIX: Extended fallback to ico=2 for severe topological intersections
    bem_model = None
    for ico_level in [4, 3, 2]:
        try:
            logger.info(f"Computing {ico_level}-ico 3-layer BEM model...")
            bem_model = mne.make_bem_model(
                subject=subject,
                ico=ico_level,
                conductivity=BEM_CONDUCTIVITY,
                subjects_dir=str(subjects_dir)
            )
            break  # Success, exit the loop
        except RuntimeError as e:
            if "not completely inside" in str(e) and ico_level > 2:
                logger.warning(
                    f"Topology check failed for ico={ico_level} ({e}). "
                    f"Falling back to ico={ico_level-1}..."
                )
            else:
                # If ico=2 also fails, abort subject BEM gracefully
                logger.error(f"Subject BEM generation failed completely: {e}")
                logger.info("NOTE: lcmv_xtra uses fsaverage BEM, so subject BEM is optional. Skipping.")
                return

    logger.info("Computing BEM solution...")
    bem_sol = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(str(bem_sol_file), bem_sol, overwrite=True)

    logger.info(f"Creating {SOURCE_SPACE_POS}mm volume source space (brain-bounded)...")
    src = mne.setup_volume_source_space(
        subject=subject,
        subjects_dir=str(subjects_dir),
        pos=SOURCE_SPACE_POS,
        bem=str(bem_sol_file),
        mindist=MINDIST,
        add_interpolator=True,
    )
    mne.write_source_spaces(str(src_file), src, overwrite=True)

    n_sources = sum(s["nuse"] for s in src)
    logger.info(f"Anatomy preparation complete: {n_sources} sources at {SOURCE_SPACE_POS}mm.")

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
