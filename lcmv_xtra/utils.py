# lcmv_xtra/utils.py

import mne
import logging
from pathlib import Path


def parse_gpsc(filepath):
    """Parse .gpsc file and return list of (name, x, y, z) tuples."""
    channels = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
            channels.append((name, x, y, z))
        except ValueError:
            continue
    return channels


def download_fsaverage(project_base, verbose=False):
    """
    Download fsaverage and generate required BEM + source space files.
    
    Parameters:
        project_base: Root BIDS directory
        verbose: Enable logging
    
    Files created:
        {project_base}/derivatives/lcmv/fsaverage/...          (from MNE)
        {project_base}/derivatives/lcmv/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif
        {project_base}/derivatives/lcmv/fsaverage/fsaverage-vol-5mm-src.fif  # ← NOW INSIDE
    """
    project_base = Path(project_base)
    lcmv_dir = project_base / 'derivatives' / 'lcmv'
    lcmv_dir.mkdir(parents=True, exist_ok=True)
    
    log = logging.getLogger('lcmv.setup')
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        log.addHandler(handler)

    fsaverage_dir = lcmv_dir / 'fsaverage'
    
    # 1. Download fsaverage if missing
    if not fsaverage_dir.exists():
        log.info("Downloading fsaverage (one-time, ~150 MB)...")
        mne.datasets.fetch_fsaverage(subjects_dir=lcmv_dir)
        log.info("✓ fsaverage downloaded.")
    else:
        log.info("fsaverage already present.")

    # 2. Generate BEM if missing
    bem_file = fsaverage_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    if not bem_file.exists():
        log.info("Generating BEM model (one-time, ~1-2 min)...")
        bem_dir = fsaverage_dir / 'bem'
        bem_dir.mkdir(exist_ok=True)
        model = mne.make_bem_model('fsaverage', ico=4, subjects_dir=lcmv_dir)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_file, bem, overwrite=True)
        log.info("✓ BEM solution saved.")
    else:
        log.info("BEM solution already exists.")

    # 3. Generate volume source space if missing → NOW SAVED INSIDE fsaverage_dir
    src_file = fsaverage_dir / 'fsaverage-vol-5mm-src.fif'  # ← KEY CHANGE
    if not src_file.exists():
        log.info("Creating 5mm volume source space...")
        src = mne.setup_volume_source_space(
            subject='fsaverage',
            subjects_dir=lcmv_dir,
            pos=5.0,
            mri='T1.mgz',
            add_interpolator=True
        )
        src.save(src_file, overwrite=True)
        log.info("✓ Volume source space saved.")
    else:
        log.info("Volume source space already exists.")

    log.info(f"✅ fsaverage resources ready at: {fsaverage_dir}")
