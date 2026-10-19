# lcmv_xtra/source_estimation_epochs_atlas_dl.py
"""
CIMT-Constrained Neural LCMV Beamformer for Epochs
====================================================
Physics-Constrained Autoencoder applied to epoched data.
The spatial filter is trained on concatenated epochs and applied to each epoch individually.
"""
import mne
import json
import logging
import lcmv_xtra
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Reuse standard epoch utilities
from .source_estimation_epochs_atlas import (
    load_epochs, validate_fsaverage, _run_coregistration, _setup_logger
)
from .source_estimation_atlas import reduce_leadfield_to_cimt

# Import Neural LCMV components
from .dl_beamformer import (
    train_beamformer, get_best_device, FINITE_EPS, N_ROIS, N_ORIENTATIONS
)

# MNE logging
mne.set_log_level('warning')
logger = logging.getLogger(__name__)


def lcmv_beamformer_epochs_atlas_dl(
    epochs: mne.Epochs,
    ch_pos: Dict,
    fsaverage_dir: Path,
    output_dir: Path,
    subject_id: str,
    task: str,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
    pick_ori: str = "max-power",
    # DL params
    nn_epochs: int = 3000,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    patience: int = 3,
    min_delta: float = 0.05,
    val_fraction: float = 0.2,
    log_interval: int = 200,
    var_loss_weight: float = 0.1,
) -> Dict:
    """
    Run epoch-based Neural LCMV source estimation with CIMT atlas reduction.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info(f"{'='*60}")
    log.info(f"Neural LCMV Epoch Atlas Source Estimation: {subject_id} - {task}")
    log.info(f"{'='*60}")

    # 1. Coregistration & Forward
    bem_file, src_file = validate_fsaverage(fsaverage_dir)
    trans_file = output_dir / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(epochs.info, ch_pos, 'fsaverage', fsaverage_dir, trans_file, log)

    src = mne.read_source_spaces(src_file)
    fwd_file = output_dir / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=n_jobs
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    # 2. Atlas Reduction (Atlas-Before-Inverse)
    log.info("Reducing lead field to 448 CIMT ROIs...")
    G_reduced, voxel_labels, _ = reduce_leadfield_to_cimt(
        fwd=fwd, src=src, verbose=verbose
    )
    
    np.save(output_dir / 'G_cimt_448.npy', G_reduced)
    np.save(output_dir / 'cimt_voxel_labels.npy', voxel_labels)

    log.info(f"Using provided epochs: {len(epochs)} trials, {epochs.tmin:.2f}s to {epochs.tmax:.2f}s")
    if len(epochs) == 0:
        raise RuntimeError("No epochs provided or all epochs were dropped.")

    epochs_eeg = epochs.copy().pick('eeg')

    # 3. Compute DATA Covariance
    log.info(f"Computing DATA covariance from window [tmin={data_cov_tmin}, tmax={data_cov_tmax}]s using method='oas'...")
    data_cov = mne.compute_covariance(
        epochs_eeg, tmin=data_cov_tmin, tmax=data_cov_tmax,
        method='oas', rank=None, n_jobs=n_jobs, verbose=False
    )

    # 4. Compute NOISE Covariance
    noise_cov = None
    if compute_noise_cov:
        log.info(f"Computing NOISE covariance from baseline [tmin={baseline_tmin}, tmax={baseline_tmax}]s using method='{noise_cov_method}'...")
        noise_cov = mne.compute_covariance(
            epochs_eeg, tmin=baseline_tmin, tmax=baseline_tmax,
            method=noise_cov_method, rank=None, n_jobs=n_jobs, verbose=False
        )

    # 5. Make Analytical LCMV filters (for initialization & orientation collapse)
    log.info("Computing Analytical LCMV filters for initialization...")
    
    fwd_temp = fwd.copy()
    fwd_temp['sol']['data'] = G_reduced
    fwd_temp['sol']['ncol'] = N_ROIS * N_ORIENTATIONS
    fwd_temp['nsource'] = N_ROIS
    fwd_temp['source_nn'] = np.tile(np.eye(N_ORIENTATIONS), (N_ROIS, 1)).astype(np.float32)
    fwd_temp['src'][0]['vertno'] = np.arange(N_ROIS, dtype=np.int32)

    filters = mne.beamformer.make_lcmv(
        info=epochs.info, forward=fwd_temp,
        data_cov=data_cov, noise_cov=noise_cov, reg=reg,
        pick_ori=pick_ori, weight_norm="unit-noise-gain", reduce_rank=True, rank=None, verbose=False
    )

    W_analytical = filters["weights"]
    max_power_ori = filters.get("max_power_ori", None)
    n_channels = G_reduced.shape[0]

    # 6. Orientation Collapse
    if pick_ori == "max-power" and max_power_ori is not None:
        G_reduced_reshaped = G_reduced.reshape(n_channels, N_ROIS, N_ORIENTATIONS)
        G_collapsed = np.einsum("csr,sr->cs", G_reduced_reshaped, max_power_ori)
        log.info("Orientation: max-power collapse via analytical weights")
    else:
        G_collapsed = G_reduced
        log.info("Orientation: vector (all %d orientations preserved)", N_ORIENTATIONS)

    # 7. Extract and concatenate epoch data for neural training
    log.info("Extracting and concatenating epoch data for neural training...")
    X_epochs = epochs.get_data(picks='eeg')  # (n_epochs, n_channels, n_times)
    n_ep, n_ch, n_times = X_epochs.shape
    
    # Concatenate along time axis: (n_channels, n_epochs * n_times)
    X_concat = X_epochs.transpose(1, 0, 2).reshape(n_ch, -1)

    # 8. Z-scoring
    log.info("Z-scoring concatenated epochs...")
    X_tensor = torch.tensor(X_concat, dtype=torch.float32)
    mean = X_tensor.mean(dim=1, keepdim=True)
    std = X_tensor.std(dim=1, keepdim=True) + FINITE_EPS
    X_z = (X_tensor - mean) / std
    G_tensor = torch.tensor(G_collapsed, dtype=torch.float32)

    # 9. Train Neural Beamformer
    log.info("Training Neural Beamformer (%d epochs)...", nn_epochs)
    W_init = torch.tensor(W_analytical, dtype=torch.float32)
    
    device = get_best_device()
    X_z = X_z.to(device)
    G_tensor = G_tensor.to(device)
    W_init = W_init.to(device)
    
    W_learned = train_beamformer(
        G_tensor, X_z, W_init,
        epochs=nn_epochs, lr=lr, betas=betas, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, patience=patience, min_delta=min_delta,
        val_fraction=val_fraction, log_interval=log_interval,
        var_loss_weight=var_loss_weight, log=log,
    )

    # 10. Apply and reshape back to epochs
    log.info("Applying learned spatial filter to epochs...")
    with torch.no_grad():
        S_concat = W_learned @ X_z  # (n_rois, n_epochs * n_times)
        
    S_concat_np = S_concat.cpu().numpy()
    
    # Reshape to (n_rois, n_epochs, n_times) then transpose to (n_epochs, n_rois, n_times)
    S_epochs = S_concat_np.reshape(N_ROIS, n_ep, n_times).transpose(1, 0, 2)

    # 11. Save STCs
    log.info(f"Saving {n_ep} epoch source estimates...")
    stcs = []
    for i in range(n_ep):
        stc_data = S_epochs[i]  # (n_rois, n_times)
        stc = mne.VolSourceEstimate(
            data=stc_data, vertices=[np.arange(N_ROIS)], tmin=epochs.tmin,
            tstep=1.0 / epochs.info['sfreq'], subject="fsaverage"
        )
        stc_file = output_dir / f'source_estimate_LCMV_epoch_{i:03d}.h5'
        stc.save(stc_file, ftype='h5', overwrite=True)
        stcs.append(stc)

    # 12. Metadata
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'sfreq_hz': float(epochs.info['sfreq']),
        'epoch_tmin': float(epochs.tmin),
        'epoch_tmax': float(epochs.tmax),
        'n_epochs': n_ep,
        'n_sources': N_ROIS,
        'n_timepoints_per_epoch': n_times,
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'data_covariance_method': 'epoch_averaged_oas',
        'data_covariance_window': [float(data_cov_tmin) if data_cov_tmin is not None else None,
                                   float(data_cov_tmax) if data_cov_tmax is not None else None],
        'compute_noise_covariance': compute_noise_cov,
        'noise_covariance_method': noise_cov_method if compute_noise_cov else None,
        'noise_baseline_window': [float(baseline_tmin) if baseline_tmin is not None else None,
                                  float(baseline_tmax) if baseline_tmax is not None else None] if compute_noise_cov else None,
        'weight_normalization': 'unit-noise-gain',
        'covariance_mode': 'separate',
        'source_space': 'CIMT_448_ROIs_Neural',
        'subject_output': str(output_dir),
        'fsaverage_dir': str(fsaverage_dir),
        'beamformer_type': 'CIMT_Constrained_Neural_LCMV_Epochs',
        'pick_ori': pick_ori,
        'nn_epochs': nn_epochs,
        'learning_rate': lr,
        'betas': list(betas),
        'weight_decay': weight_decay,
        'grad_clip_norm': grad_clip_norm,
        'patience': patience,
        'min_delta': min_delta,
        'val_fraction': val_fraction,
        'log_interval': log_interval,
        'var_loss_weight': var_loss_weight,
    }
    with open(output_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Neural Epoch atlas source estimation complete: {output_dir}")
    log.info(f"{'='*60}\n")
    return metadata


def execute_source_estimation_epochs_atlas_dl(
    project_base: Path,
    subject_id: str,
    task: str,
    epochs_file_path: str,
    fsaverage_dir: Path,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
    pick_ori: str = "max-power",
    nn_epochs: int = 3000,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    patience: int = 3,
    min_delta: float = 0.05,
    val_fraction: float = 0.2,
    log_interval: int = 200,
    var_loss_weight: float = 0.1,
) -> Dict:
    """High-level orchestrator for epoch-based Neural LCMV source estimation."""
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'

    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")

    epochs_full_path = project_base / epochs_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_epochs_atlas_dl'

    epochs, ch_pos = load_epochs(
        epochs_file_path=epochs_full_path, gpsc_file_path=gpsc_full_path,
        subject_id=subject_id, logger=None
    )

    return lcmv_beamformer_epochs_atlas_dl(
        epochs=epochs, ch_pos=ch_pos, fsaverage_dir=fsaverage_dir, output_dir=output_dir,
        subject_id=subject_id, task=task,
        reg=reg, n_jobs=n_jobs, verbose=verbose,
        noise_cov_method=noise_cov_method,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        data_cov_tmin=data_cov_tmin, data_cov_tmax=data_cov_tmax,
        compute_noise_cov=compute_noise_cov,
        pick_ori=pick_ori,
        nn_epochs=nn_epochs, lr=lr, betas=betas, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, patience=patience, min_delta=min_delta,
        val_fraction=val_fraction, log_interval=log_interval,
        var_loss_weight=var_loss_weight,
    )
