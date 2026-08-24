# LCMV Source Reconstruction

Linearly Constrained Minimum Variance (LCMV) beamforming for high-density EEG source reconstruction. This library provides a fully automated pipeline optimized for the BEL 280-channel system, featuring atlas-constrained inverse modeling, physics-informed neural beamforming, and batch-ready tensor assembly.

<br>

#### Key Features

- **BEL 280 Coregistration**: Automated alignment using bundled `.gpsc` digitized electrode coordinates with ICP refinement and outlier rejection.
- **Three Source Estimation Strategies**:
  - *Standard*: Full volumetric LCMV + post-hoc atlas extraction.
  - *Atlas-Constrained*: Forward model reduced to 448 CIMT ROIs *before* inversion (prevents intra-ROI cancellation).
  - *Neural (PyTorch)*: Physics-constrained autoencoder that refines analytical LCMV weights via reconstruction + variance loss.
- **Multi-Atlas Support**: CIMT Unified (448 ROIs), Glasser+Tian (414 ROIs), DiFuMo (512 components), and custom MNI coordinates.
- **Batch Tensor Assembly**: Parallel processing for multi-subject studies with automatic resampling and stacking into 3D/4D tensors.
- **Connectivity Analysis**: WPLI2-debiased spectral connectivity for motor-basal-executive networks.
- **Self-Contained**: All atlas templates, channel montages, and fsaverage setup utilities bundled in the package.

> **Note**: Optimized for BEL 280-channel EEG with `fsaverage`, but adaptable to other high-density setups.

<br>

## Installation

### Core (Analytical Pipelines Only)
```bash
pip install git+https://github.com/cimt-unia/lcmv_xtra.git
```

### With Neural Beamformer Support
The neural beamformer requires PyTorch and is installed as an optional dependency:
```bash
pip install "git+https://github.com/cimt-unia/lcmv_xtra.git[ml]"
```

### Jupyter / Colab
```python
!pip install --user "git+https://github.com/cimt-unia/lcmv_xtra.git[ml]"
```

<br>

## Quick Start

### Step 1: One-Time Project Setup
```python
from lcmv_xtra import download_fsaverage

fs_dir = '/path/to/fsaverage'
download_fsaverage(fs_dir)  # Downloads anatomy, generates BEM + 5mm volume source space
```

### Step 2: Source Estimation

Choose the strategy that best fits your analysis:

#### Option A: Atlas-Constrained LCMV (Recommended)
Reduces the forward model to 448 CIMT ROIs *before* computing beamformer weights. Output is directly `(448, T)` — no post-hoc extraction needed.
```python
from lcmv_xtra import execute_source_estimation_atlas

metadata = execute_source_estimation_atlas(
    project_base='/path/to/bids_root',
    subject_id='sub-01',
    task='move',
    ica_file_path='derivatives/ica/sub-01_task-move_ica.fif',
    fsaverage_dir=fs_dir,
)
# Output: source_estimate_LCMV.h5 with shape (448, n_times)
```

#### Option B: Neural Beamformer (PyTorch)
Refines analytical LCMV weights using a physics-constrained autoencoder. Requires `[ml]` install.
```python
from lcmv_xtra import execute_source_estimation_atlas_pytorch

metadata = execute_source_estimation_atlas_pytorch(
    project_base='/path/to/bids_root',
    subject_id='sub-01',
    task='move',
    ica_file_path='derivatives/ica/sub-01_task-move_ica.fif',
    fsaverage_dir=fs_dir,
    nn_epochs=20000,  # Early stopping enabled (patience=10)
)
# Saves both W_analytical.npy and W_neural_learned.npy for comparison
```

#### Option C: Standard LCMV + Post-Hoc Extraction
Full volumetric source estimation followed by atlas-based time course extraction.
```python
from lcmv_xtra import execute_source_estimation, cimt_extraction

metadata = execute_source_estimation(
    project_base='/path/to/bids_root',
    subject_id='sub-01',
    task='move',
    ica_file_path='derivatives/ica/sub-01_task-move_ica.fif',
    fsaverage_dir=fs_dir,
)

cimt_tc, cimt_labels = cimt_extraction(
    subject_output_dir=metadata['subject_output'],
    fsaverage_dir=metadata['fsaverage_dir'],
)
```

### Step 3: Batch Tensor Assembly
Process multiple subjects in parallel and stack into a single tensor:
```python
import lcmv_xtra as lx

df = lx.scan_eeg_paths('/path/to/cleaned/', pattern="*_rest_cleaned.fif")

# Atlas-constrained tensor (448 ROIs, no post-hoc extraction)
lx.assemble_atlas_tensor(df, fs_dir, output_dir, task_name="rest", n_jobs=-1)

# Or standard tensor with post-hoc CIMT extraction
lx.assemble_tensor(df, fs_dir, output_dir, task_name="rest", n_jobs=-1)
```

### Step 4: Connectivity Analysis
```python
from lcmv_xtra import compute_cimt_full_connectivity, compute_cimt_motor_connectivity

# Full 448×448 connectivity matrix
full_conn = compute_cimt_full_connectivity(cimt_tc, band_name="beta")

# Reduced motor-basal-executive-STN network
motor_conn = compute_cimt_motor_connectivity(cimt_tc, band_name="beta")
```

#### Supported Frequency Bands
| Band | Range (Hz) |
|------|-----------|
| `theta` | 4–8 |
| `alpha` | 8–12 |
| `low_beta` | 13–20 |
| `high_beta` | 20–30 |
| `beta` | 13–30 |
| `low_gamma` | 30–60 |
| `high_gamma` | 60–120 |
| `gamma` | 30–100 |

<br>

## Technical Details

### Coregistration
Three-stage procedure with explicit failure on poor alignment:
1. **Fiducial alignment** (`FidNz`, `FidT9`, `FidT10`)
2. **ICP refinement** using all 280 channels as head shape points (nasion weight = 2.0), with outlier rejection (>5 mm MRI distance)
3. **Final ICP refinement** (nasion weight = 10.0, 20 iterations)

Pipeline **fails explicitly** if mean coregistration error exceeds 5 mm. Typical mean error: **~4.86 mm**.

### Source Estimation Parameters
- **Source space**: 5-mm isotropic volume grid bounded by inner skull
- **Forward model**: Three-shell BEM (`fsaverage-5120-5120-5120-bem-sol.fif`)
- **Covariance**: OAS-regularized estimate from entire recording (or per-epoch for epoch pipelines)
- **Beamformer**: Max-power orientation, unit-noise-gain normalization, reduced rank, Tikhonov regularization (λ = 0.05)
- **Neural refinement**: Smooth L1 reconstruction loss + VICReg variance anti-collapse term, AdamW/ScheduleFree optimizer, early stopping

### Connectivity
- **Estimator**: WPLI2 debiased with multitaper spectral analysis
- **Output**: Symmetric matrices with ROI-labeled indices/columns

<br>

## 📁 Expected Project Layout

```
bids_root/
├── derivatives/
│   ├── ica/                              # ICA-cleaned FIF files (input)
│   └── lcmv/                             # Auto-generated by download_fsaverage()
│       ├── fsaverage/                    # Full fsaverage anatomy + BEM
│       │   ├── bem/
│       │   │   └── fsaverage-5120-5120-5120-bem-sol.fif
│       │   └── mri/T1.mgz
│       └── fsaverage-vol-5mm-src.fif     # Precomputed volume source space
└── sub-01/
    └── eeg/
        └── sub-01_task-move_eeg.fif      # Raw data (if needed)

```

> 💡 The `derivatives/lcmv/` directory is **created and populated automatically** by `download_fsaverage`. All source estimates are saved in `.h5` format for efficiency.


<br>

### Summary of Changes

| Section | Change | Rationale |
|---------|--------|-----------|
| **Header** | Added neural beamformer and batch tensor assembly to feature list | Reflects current library capabilities |
| **Installation** | Split into core vs `[ml]` variants | Matches new `pyproject.toml` optional dependencies |
| **Quick Start** | Restructured into 3 source estimation options (A/B/C) | Atlas-constrained is now the recommended default; neural is prominently featured |
| **Batch Assembly** | Added Step 3 with `assemble_atlas_tensor` | Critical workflow that was missing from README |
| **Technical Details** | Added neural refinement parameters | Documents loss function, optimizer, and early stopping |
| **Connectivity** | Simplified to CIMT examples | Reduces clutter; GT/DiFuMo still available but not primary |
| **Frequency Bands** | Converted to table | More scannable than bullet list |
