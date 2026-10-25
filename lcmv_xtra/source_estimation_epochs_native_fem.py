# lcmv_xtra/source_estimation_epochs_native_fem.py

import mne
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Import shared utilities from the standard epochs module
from .source_estimation_epochs import (
    load_epochs, _setup_logger
)

mne.set_log_level('warning')

def _remove_outliers(coreg: mne.coreg.Coregistration, threshold_mm: float, log: logging.Logger) -> None:
    """Helper to compute distances and omit head shape points beyond a threshold."""
    dists = coreg.compute_dig_mri_distances()
    n_outliers = np.sum(dists > threshold_mm / 1000.0)
    if n_outliers > 0:
        log.info(f" -> Removing {n_outliers} outliers > {threshold_mm}mm...")
        coreg.omit_head_shape_points(distance=threshold_mm / 1000.0)

def _run_coregistration_native(info: mne.Info, ch_pos: Dict, subject: str,
                               subjects_dir: Path, trans_file: Path,
                               logger: logging.Logger) -> Tuple[mne.transforms.Transform, Dict]:
    """Run optimized native-space rigid-body coregistration with multi-stage ICP."""
    log = logger or logging.getLogger(__name__)
    coreg = mne.coreg.Coregistration(
        info, subject=subject, subjects_dir=subjects_dir, fiducials='estimated' 
    )
    log.info("Fitting fiducials...")
    coreg.fit_fiducials(verbose=False)
    log.info("Running Coarse ICP (20 iters, nasion_weight=1.0)...")
    coreg.fit_icp(n_iterations=20, nasion_weight=1.0, verbose=False)
    _remove_outliers(coreg, threshold_mm=8.0, log=log)
    log.info("Running Fine ICP (50 iters, nasion_weight=5.0)...")
    coreg.fit_icp(n_iterations=50, nasion_weight=5.0, verbose=False)
    _remove_outliers(coreg, threshold_mm=10.0, log=log)
    log.info("Running final polish ICP (10 iters)...")
    coreg.fit_icp(n_iterations=10, nasion_weight=5.0, verbose=False)

    dists_mm = coreg.compute_dig_mri_distances() * 1000
    mean_err, median_err, max_err = np.mean(dists_mm), np.median(dists_mm), np.max(dists_mm)
    log.info(f"Coreg error (mm) - Mean: {mean_err:.2f}, Median: {median_err:.2f}, Max: {max_err:.2f}")

    if mean_err > 8.0:
        raise RuntimeError(f"Mean error {mean_err:.2f}mm exceeds 8.0mm threshold.")
    elif mean_err > 5.0:
        log.warning(f"Mean error {mean_err:.2f}mm is > 5.0mm. Verify visually.")
    else:
        log.info(f"Coregistration successful: Mean error {mean_err:.2f}mm is excellent.")

    mne.write_trans(str(trans_file), coreg.trans, overwrite=True)
    return coreg.trans, {'mean': mean_err, 'median': median_err, 'max': max_err}
                                   
def validate_fem_anatomy(subject_id: str, simnibs_dir: Path) -> Tuple[Path, Path]:
    """Validate SimNIBS FEM forward solution and source space."""
    fem_dir = simnibs_dir / f"fem_{subject_id}"
    fwd_file = fem_dir / f"{subject_id}_leadfield_eeg_montage-fwd.fif"
    src_file = fem_dir / f"{subject_id}_leadfield_eeg_montage-src.fif"
    
    if not (fwd_file.exists() and src_file.exists()):
        raise FileNotFoundError(
            f"Missing FEM files for {subject_id} in {fem_dir}. "
            "Please run 02b_simnibs_fem_and_export.py first."
        )
    return fwd_file, src_file
         
def lcmv_beamformer_epochs_native_fem(
    epochs: mne.Epochs,
    ch_pos: Dict,
    subjects_dir: Path,
    simnibs_dir: Path,
    output_dir: Path,
    subject_id: str,
    task: str,
    reg: float = 0.05,
    mindist: float = 2.0, 
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
    coregister: bool = True,
    trans_file: Optional[Path] = None,
) -> Dict:
    """
    Run epoch-based LCMV source estimation using a precomputed SimNIBS FEM forward model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(subject_id, task, output_dir, verbose)
    
    log.info(f"{'='*60}")
    log.info(f"FEM-BASED LCMV Epoch Source Estimation: {subject_id} - {task}")
    log.info(f"{'='*60}")

    # 1. Load Precomputed FEM Forward Solution (With Cross-Version Workaround)
    fem_fwd_file, _ = validate_fem_anatomy(subject_id, simnibs_dir)
    log.info(f"Loading precomputed FEM forward solution from {fem_fwd_file}...")
    
    try:
        fwd = mne.read_forward_solution(str(fem_fwd_file))
    except ValueError as e:
        if "MRI/head" in str(e):
            log.warning("Forward file missing MRI/head transform. Applying robust workaround...")
            # Load the transform from the saved file
            trans_path = simnibs_dir / "head_ras-trans.fif"
            trans = mne.read_trans(str(trans_path))
            mri_head_t = mne.transforms.invert_transform(trans)
            
            # Monkey-patch the reader to bypass the strict check safely
            import mne.forward.forward as _fwd_mod
            _orig = _fwd_mod._read_forward_meas_info
            def _patched(tree, fid):
                try:
                    return _orig(tree, fid)
                except ValueError:
                    info_path = simnibs_dir / "montage_info.fif"
                    info = mne.io.read_info(str(info_path))
                    if 'dev_head_t' not in info or info['dev_head_t'] is None:
                        info['dev_head_t'] = mne.transforms.Transform('meg', 'head', np.eye(4))
                    return info
                    
            _fwd_mod._read_forward_meas_info = _patched
            try:
                fwd = mne.read_forward_solution(str(fem_fwd_file))
            finally:
                _fwd_mod._read_forward_meas_info = _orig
            
            # ⚠️ FIX: Inject ONLY at the top level.
            # MNE's Info object blocks direct assignment to info['mri_head_t'].
            fwd['mri_head_t'] = mri_head_t
            log.info("Successfully injected MRI/head transform via workaround.")
        else:
            raise
    
    # Extract source space directly from the forward object
    src_native = fwd['src'] 

    # 2. Coregistration (Run purely for metadata/error tracking)
    if trans_file is None:
        trans_file = output_dir / f'{subject_id}-trans.fif'
    
    if coregister:
        log.info("Running ICP coregistration to native MRI (for metadata)...")
        trans, coreg_errors = _run_coregistration_native(
            epochs.info, ch_pos, subject_id, subjects_dir, trans_file, log
        )
    else:
        log.info("Skipping coregistration (coregister=False).")
        if trans_file.exists():
            trans = mne.read_trans(str(trans_file))
            coreg_errors = {'mean': 0.0, 'median': 0.0, 'max': 0.0}
        else:
            log.warning("No trans_file provided. Using identity transform for metadata.")
            trans = mne.transforms.Transform('head', 'mri')
            coreg_errors = {'mean': 0.0, 'median': 0.0, 'max': 0.0}

    # 3. Compute Covariances
    epochs_eeg = epochs.copy().pick('eeg')

    log.info(f"Computing DATA covariance from window [{data_cov_tmin}, {data_cov_tmax}]s...")
    data_cov = mne.compute_covariance(
        epochs_eeg, tmin=data_cov_tmin, tmax=data_cov_tmax,
        method='oas', rank=None, n_jobs=n_jobs, verbose=False
    )
    
    noise_cov = None
    if compute_noise_cov:
        log.info(f"Computing NOISE covariance from baseline [{baseline_tmin}, {baseline_tmax}]s...")
        noise_cov = mne.compute_covariance(
            epochs_eeg, tmin=baseline_tmin, tmax=baseline_tmax,
            method=noise_cov_method, rank=None, n_jobs=n_jobs, verbose=False
        )

    # 4. Compute LCMV Filters & Apply
    log.info("Computing LCMV filters in native space...")
    filters = mne.beamformer.make_lcmv(
        info=epochs.info, forward=fwd,
        data_cov=data_cov, noise_cov=noise_cov, reg=reg,
        pick_ori='max-power', weight_norm='unit-noise-gain',
        reduce_rank=True, rank=None, verbose=False
    )

    log.info("Applying LCMV beamformer to single-trial epochs...")
    stcs_native: List[mne.VolSourceEstimate] = mne.beamformer.apply_lcmv_epochs(
        epochs=epochs, filters=filters
    )

    # 5. Save Outputs
    native_dir = output_dir / "native_stcs"
    native_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving {len(stcs_native)} UN-MORPHED native epoch STCs to {native_dir}...")
    for i, stc_nat in enumerate(stcs_native):
        native_file = native_dir / f'source_estimate_LCMV_epoch_{i:03d}_native.h5'
        stc_nat.save(native_file, ftype='h5', overwrite=True)

    metadata = {
        'subject_id': subject_id,
        'task': task,
        'space': 'pure_native_unmorphed',
        'forward_model_type': 'SimNIBS_FEM',
        'coregistration_applied': coregister,
        'forward_mindist_mm': float(mindist),
        'sfreq_hz': float(epochs.info['sfreq']),
        'epoch_tmin': float(epochs.tmin),
        'epoch_tmax': float(epochs.tmax),
        'n_epochs': len(stcs_native),
        'n_sources': int(stcs_native[0].data.shape[0]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'data_covariance_window': [data_cov_tmin, data_cov_tmax],
        'compute_noise_covariance': compute_noise_cov,
        'noise_baseline_window': [baseline_tmin, baseline_tmax] if compute_noise_cov else None,
        'weight_normalization': 'unit-noise-gain',
        'covariance_mode': 'separate',
        'subject_output': str(output_dir),
        'native_stcs_dir': str(native_dir),
    }
    with open(output_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"FEM native epoch source estimation complete: {output_dir}")
    log.info(f"{'='*60}\n")
    return metadata

def execute_source_estimation_epochs_native_fem(
    project_base: Path,
    subject_id: str,
    task: str,
    epochs_file_path: str,
    subjects_dir: Path,
    simnibs_dir: Path = Path("/mnt/movement/users/jaizor/xtra/simnibs"),
    reg: float = 0.05,
    mindist: float = 2.0,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
    coregister: bool = True,
    trans_file: Optional[Path] = None,
) -> Dict:
    """High-level orchestrator for FEM-based epoch LCMV source estimation."""
    import lcmv_xtra
    
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig_old.gpsc'

    epochs_full_path = project_base / epochs_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_epochs_native_fem'

    epochs, ch_pos = load_epochs(
        epochs_file_path=epochs_full_path, gpsc_file_path=gpsc_full_path,
        subject_id=subject_id, logger=None
    )

    return lcmv_beamformer_epochs_native_fem(
        epochs=epochs, ch_pos=ch_pos, 
        subjects_dir=subjects_dir, 
        simnibs_dir=simnibs_dir,
        output_dir=output_dir, subject_id=subject_id, task=task,
        reg=reg, mindist=mindist, n_jobs=n_jobs, verbose=verbose, 
        noise_cov_method=noise_cov_method,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        data_cov_tmin=data_cov_tmin, data_cov_tmax=data_cov_tmax,
        compute_noise_cov=compute_noise_cov,
        coregister=coregister,
        trans_file=trans_file,
    )
