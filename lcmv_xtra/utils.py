# lcmv_xtra/utils.py

import mne
import logging
from pathlib import Path


def parse_gpsc(filepath):
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


def download_fsaverage(target_dir, verbose=False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger('lcmv.setup')
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        log.addHandler(handler)

    fsaverage_dir = target_dir / 'fsaverage'

    if not fsaverage_dir.exists():
        log.info("Downloading fsaverage (one-time, ~150 MB)...")
        mne.datasets.fetch_fsaverage(subjects_dir=target_dir)
        log.info("✓ fsaverage downloaded.")
    else:
        log.info("fsaverage already present.")

    bem_file = fsaverage_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    if not bem_file.exists():
        log.info("Generating BEM model (one-time, ~1-2 min)...")
        bem_dir = fsaverage_dir / 'bem'
        bem_dir.mkdir(exist_ok=True)
        model = mne.make_bem_model('fsaverage', ico=4, subjects_dir=target_dir)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_file, bem, overwrite=True)
        log.info("✓ BEM solution saved.")
    else:
        log.info("BEM solution already exists.")

    src_file = target_dir / 'fsaverage-vol-5mm-src.fif'

    needs_regeneration = False

    if not src_file.exists():
        needs_regeneration = True
        log.info("Creating 5mm volume source space bounded by inner skull...")
    else:
        src_check = mne.read_source_spaces(str(src_file))
        n_sources = len(src_check[0]['vertno'])
        if n_sources >= 20000:
            log.warning(f"⚠️  Existing source space has {n_sources} sources (likely old 90mm sphere).")
            log.warning("   Regenerating with proper inner-skull bounding...")
            needs_regeneration = True
        else:
            log.info(f"✓ Volume source space already exists ({n_sources} brain-bounded sources).")

    if needs_regeneration:
        src = mne.setup_volume_source_space(
            subject='fsaverage',
            subjects_dir=target_dir,
            pos=5.0,
            mri='T1.mgz',
            bem=bem_file,
            mindist=5.0,
            add_interpolator=True
        )
        src.save(src_file, overwrite=True)
        log.info(f"✓ Volume source space saved ({len(src[0]['vertno'])} brain-bounded sources).")

    log.info(f"✅ fsaverage resources ready at: {target_dir}")
