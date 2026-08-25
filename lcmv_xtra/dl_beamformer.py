# lcmv_xtra/dl_beamformer.py
"""
CIMT-Constrained Neural LCMV Beamformer
================================================
Physics-Constrained Autoencoder. Loss = Reconstruction + Variance.
No batching (fits in GPU). No tqdm.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from schedulefree import AdamWScheduleFree
    HAS_SCHEDULE_FREE = True
except ImportError:
    HAS_SCHEDULE_FREE = False

from .source_estimation import (
    load_subject,
    validate_fsaverage,
    _run_coregistration,
    _setup_logger,
)
from .source_estimation_atlas import reduce_leadfield_to_cimt

import lcmv_xtra

# Constants (defaults — all overridable via function parameters)
N_ROIS = 448
N_ORIENTATIONS = 3
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
EPOCHS = 3000
PATIENCE = 3
MIN_DELTA = 0.05
VAL_FRACTION = 0.2
LOG_INTERVAL = 200
FINITE_EPS = 1e-8
VAR_LOSS_WEIGHT = 0.1

logger = logging.getLogger(__name__)


def get_best_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, weights):
        if self.best_score is None or val_loss < (self.best_score - self.min_delta):
            self.best_score = val_loss
            self.best_weights = weights.detach().clone()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class NeuralSpatialFilter(nn.Module):
    """Physics-constrained neural beamformer. One trainable matrix W."""

    def __init__(self, G_z, W_init):
        super().__init__()
        self.W = nn.Parameter(W_init.clone())
        self.register_buffer("G_z", G_z)

    def forward(self, X_z):
        S_hat = self.W @ X_z
        X_hat = self.G_z @ S_hat
        return S_hat, X_hat


def compute_loss(X_z, X_hat, S_hat, var_loss_weight=VAR_LOSS_WEIGHT):
    """Reconstruction + Variance (anti-collapse)."""
    recon_loss = F.smooth_l1_loss(X_z, X_hat, beta=1.0)

    std_s = torch.sqrt(S_hat.var(dim=1) + FINITE_EPS)
    var_loss = F.relu(1.0 - std_s).mean()

    return recon_loss + (var_loss_weight * var_loss)


def train_beamformer(G_z, X_z, W_init, epochs=EPOCHS, lr=LEARNING_RATE,
                     betas=BETAS, weight_decay=WEIGHT_DECAY,
                     grad_clip_norm=GRAD_CLIP_NORM,
                     patience=PATIENCE, min_delta=MIN_DELTA,
                     val_fraction=VAL_FRACTION, log_interval=LOG_INTERVAL,
                     var_loss_weight=VAR_LOSS_WEIGHT,
                     log: Optional[logging.Logger] = None):
    """Train with reconstruction + variance loss. No batching."""
    if log is None:
        log = logger

    device = X_z.device

    val_start = int(X_z.shape[1] * (1.0 - val_fraction))
    X_train = X_z[:, :val_start]
    X_val = X_z[:, val_start:]

    log.info("Train: %d | Val: %d | Device: %s", X_train.shape[1], X_val.shape[1], device)
    log.info("LR: %.2e | Betas: %s | Weight Decay: %.2e | Grad Clip: %.1f",
             lr, betas, weight_decay, grad_clip_norm)
    log.info("Patience: %d | Min delta: %.4f | Val fraction: %.2f | Log every: %d epochs",
             patience, min_delta, val_fraction, log_interval)
    log.info("Var loss weight: %.3f", var_loss_weight)

    model = NeuralSpatialFilter(G_z, W_init).to(device)

    if HAS_SCHEDULE_FREE:
        optimizer = AdamWScheduleFree(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        model.train()
        if HAS_SCHEDULE_FREE:
            optimizer.train()

        optimizer.zero_grad(set_to_none=True)

        S_hat, X_hat = model(X_train)
        loss = compute_loss(X_train, X_hat, S_hat, var_loss_weight=var_loss_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        # --- Early stopping evaluation (EVERY epoch) ---
        model.eval()
        if HAS_SCHEDULE_FREE:
            optimizer.eval()

        with torch.no_grad():
            S_val, X_val_hat = model(X_val)
            val_loss = compute_loss(X_val, X_val_hat, S_val, var_loss_weight=var_loss_weight)

        early_stopper(val_loss.item(), model.W)

        if early_stopper.early_stop:
            log.info("Early stopping at epoch %d.", epoch + 1)
            break

        # --- Console logging (every log_interval epochs) ---
        if (epoch + 1) % log_interval == 0:
            log.info("Epoch %04d/%d | Train: %.4e | Val: %.4e",
                     epoch + 1, epochs, loss.item(), val_loss.item())

        model.train()
        if HAS_SCHEDULE_FREE:
            optimizer.train()

    if early_stopper.best_weights is not None:
        log.info("Using best weights from early stopping.")
        return early_stopper.best_weights.to(device)

    return model.W.detach()


def lcmv_beamformer_cimt_pytorch(input, ch_pos, fsaverage_dir, output_dir,
                                 subject_id, task, reg=0.05, n_jobs=1,
                                 pick_ori="max-power",
                                 nn_epochs=EPOCHS, lr=LEARNING_RATE,
                                 betas=BETAS, weight_decay=WEIGHT_DECAY,
                                 grad_clip_norm=GRAD_CLIP_NORM,
                                 patience=PATIENCE, min_delta=MIN_DELTA,
                                 val_fraction=VAL_FRACTION,
                                 log_interval=LOG_INTERVAL,
                                 var_loss_weight=VAR_LOSS_WEIGHT,
                                 verbose=False):
    """Run the full CIMT-constrained neural LCMV pipeline."""
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info("CIMT-Constrained Neural LCMV: %s - %s", subject_id, task)

    bem_file, src_file = validate_fsaverage(fsaverage_dir)

    log.info("Coregistration...")
    trans_file = output_dir / "fsaverage-trans.fif"
    trans, coreg_errors = _run_coregistration(input, ch_pos, "fsaverage", fsaverage_dir, trans_file, log)

    log.info("Forward solution...")
    src = mne.read_source_spaces(src_file)
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(input.info, trans=trans, src=src, bem=bem,
                                    eeg=True, mindist=5.0, n_jobs=n_jobs)

    log.info("Atlas reduction...")
    G_reduced, voxel_labels, _ = reduce_leadfield_to_cimt(fwd=fwd, src=src, verbose=True)

    log.info("Analytical LCMV (pick_ori=%s)...", pick_ori)
    cov = mne.compute_raw_covariance(input, method="oas", picks="eeg", rank=None, n_jobs=n_jobs, verbose=False)

    fwd_temp = fwd.copy()
    fwd_temp["sol"]["data"] = G_reduced
    fwd_temp["sol"]["ncol"] = N_ROIS * N_ORIENTATIONS
    fwd_temp["nsource"] = N_ROIS
    fwd_temp["source_nn"] = np.tile(np.eye(N_ORIENTATIONS), (N_ROIS, 1)).astype(np.float32)
    fwd_temp["src"][0]["vertno"] = np.arange(N_ROIS, dtype=np.int32)

    filters = mne.beamformer.make_lcmv(
        info=input.info, forward=fwd_temp, data_cov=cov, noise_cov=cov, reg=reg,
        pick_ori=pick_ori, weight_norm="unit-noise-gain", reduce_rank=True, rank=None, verbose=False,
    )

    W_analytical = filters["weights"]
    max_power_ori = filters.get("max_power_ori", None)

    n_channels = G_reduced.shape[0]

    if pick_ori == "max-power" and max_power_ori is not None:
        # Collapse orientations using max-power orientation weights
        G_reduced_reshaped = G_reduced.reshape(n_channels, N_ROIS, N_ORIENTATIONS)
        G_collapsed = np.einsum("csr,sr->cs", G_reduced_reshaped, max_power_ori)
        log.info("Orientation: max-power collapse via analytical weights")
    else:
        # Use vector beamformer output directly (3 orientations per ROI)
        G_collapsed = G_reduced
        log.info("Orientation: vector (all %d orientations preserved)", N_ORIENTATIONS)

    log.info("Z-scoring...")
    X_raw = input.get_data(picks="eeg")
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    G_tensor = torch.tensor(G_collapsed, dtype=torch.float32)

    mean = X_tensor.mean(dim=1, keepdim=True)
    std = X_tensor.std(dim=1, keepdim=True) + FINITE_EPS
    X_z = (X_tensor - mean) / std
    G_z = G_tensor  # Do NOT divide G by std

    W_init = torch.tensor(W_analytical, dtype=torch.float32)

    device = get_best_device()
    X_z = X_z.to(device)
    G_z = G_z.to(device)
    W_init = W_init.to(device)

    log.info("Training (%d epochs)...", nn_epochs)
    W_learned = train_beamformer(
        G_z, X_z, W_init,
        epochs=nn_epochs, lr=lr, betas=betas, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, patience=patience, min_delta=min_delta,
        val_fraction=val_fraction, log_interval=log_interval,
        var_loss_weight=var_loss_weight, log=log,
    )

    log.info("Applying...")
    with torch.no_grad():
        S_hat = W_learned @ X_z

    stc_data = S_hat.cpu().numpy()

    log.info("Saving...")
    np.save(output_dir / "G_cimt_448_collapsed.npy", G_collapsed)
    np.save(output_dir / "cimt_voxel_labels.npy", voxel_labels)
    np.save(output_dir / "W_analytical.npy", W_analytical)
    np.save(output_dir / "W_neural_learned.npy", W_learned.cpu().numpy())
    np.save(output_dir / "source_estimate_neural.npy", stc_data)

    stc = mne.VolSourceEstimate(
        data=stc_data, vertices=[np.arange(N_ROIS)], tmin=0.0,
        tstep=1.0 / input.info["sfreq"], subject="fsaverage",
    )
    stc.save(output_dir / "source_estimate_LCMV.h5", ftype="h5", overwrite=True)

    metadata = {
        "subject_id": subject_id,
        "task": task,
        "sfreq_hz": float(input.info["sfreq"]),
        "duration_min": float(input.n_times / input.info["sfreq"] / 60),
        "n_sources": N_ROIS,
        "n_timepoints": int(stc_data.shape[1]),
        "coreg_mean_error_mm": float(coreg_errors["mean"]),
        "regularization": reg,
        "pick_ori": pick_ori,
        "source_space": "CIMT_448_ROIs_Neural",
        "subject_output": str(output_dir),
        "fsaverage_dir": str(fsaverage_dir),
        "beamformer_type": "CIMT_Constrained_Neural_LCMV",
        "nn_epochs": nn_epochs,
        "learning_rate": lr,
        "betas": list(betas),
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "patience": patience,
        "min_delta": min_delta,
        "val_fraction": val_fraction,
        "log_interval": log_interval,
        "var_loss_weight": var_loss_weight,
        "optimizer": "AdamWScheduleFree" if HAS_SCHEDULE_FREE else "AdamW",
        "loss_components": ["SmoothL1_Reconstruction", "VICReg_Variance"],
    }
    with open(output_dir / "pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Complete: %s", output_dir)
    return metadata


def execute_source_estimation_atlas_pytorch(
    project_base, subject_id, task, ica_file_path, fsaverage_dir,
    reg=0.05, n_jobs=1, pick_ori="max-power",
    nn_epochs=EPOCHS, lr=LEARNING_RATE,
    betas=BETAS, weight_decay=WEIGHT_DECAY,
    grad_clip_norm=GRAD_CLIP_NORM,
    patience=PATIENCE, min_delta=MIN_DELTA,
    val_fraction=VAL_FRACTION, log_interval=LOG_INTERVAL,
    var_loss_weight=VAR_LOSS_WEIGHT, verbose=False,
):
    """High-level orchestrator."""
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / "data" / "bel_280" / "ghw280_from_egig.gpsc"

    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")

    ica_full_path = project_base / ica_file_path
    output_dir = project_base / "derivatives" / "lcmv" / f"{subject_id}_{task}_cimt_neural"

    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path, gpsc_file_path=gpsc_full_path,
        subject_id=subject_id, logger=None,
    )

    return lcmv_beamformer_cimt_pytorch(
        input=raw, ch_pos=ch_pos, fsaverage_dir=fsaverage_dir,
        output_dir=output_dir, subject_id=subject_id, task=task,
        reg=reg, n_jobs=n_jobs, pick_ori=pick_ori,
        nn_epochs=nn_epochs, lr=lr, betas=betas, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, patience=patience, min_delta=min_delta,
        val_fraction=val_fraction, log_interval=log_interval,
        var_loss_weight=var_loss_weight, verbose=verbose,
    )

"""
lx.assemble_dl_tensor_epochs(
    data_index=pre_df,
    fs_dir=FS_DIR,
    output_dir=OUTPUT_DIR,
    task_name="sm_pre",
    project_base=PROJECT_BASE,
    epoch_duration=2.5,
    target_sfreq=250.0,
    n_jobs=1,
    # All beamformer params now configurable:
    nn_epochs=8000,
    lr=5e-5,
    patience=5,
    min_delta=0.02,
    log_interval=100,
    pick_ori="max-power",
    var_loss_weight=0.05,
    verbose=True,
)
"""
