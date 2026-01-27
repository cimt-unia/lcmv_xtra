# LCMV Source Reconstruction

A robust, reproducible pipeline for LCMV beamformer-based source estimation on high-density EEG (e.g., BEL 280-channel systems), featuring:

- **BEL 280 coregistration** using bundled `.gpsc` digitized electrode coordinates
- **Atlas-based time course extraction**: DiFuMo (512 components) and Glasser+Tian (414 ROIs)
- **Optimized for fsaverage** with precomputed BEM and volume source space
- **Batch-ready** for multi-subject processing
- **Self-contained**: All required data files bundled in package

> **Note**: Optimized for BEL 280-channel EEG with `fsaverage`, but adaptable to other high-density setups.

---

## 📦 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/cimt-unia/lcmv_xtra.git
!python -m pip install --user git+https://github.com/cimt-unia/lcmv_xtra.git
```

> Requires pre-downloaded `fsaverage` in your project's `derivatives/lcmv/` directory.

---

## Usage

### Single Subject
```python
from lcmv_xtra import execute_source_estimation, difumo_extraction, gt_extraction

# Step 1: Source Estimation
metadata = execute_source_estimation(
    project_base='/path/to/bids',
    subject_id='sub-01',
    task='move',
    ica_file_path='derivatives/ica/sub-01_task-move_ica.fif'
)


# Step 2: Glasser & Tian Atlas 
gt_tc, _ = gt_extraction(
    subject_output_dir=metadata['subject_output'],
    global_subjects_dir=metadata['global_subjects_dir']
)


# Step 3: Difumo Atlas 
difumo_tc, _ = difumo_extraction(
    subject_output_dir=metadata['subject_output'],
    global_subjects_dir=metadata['global_subjects_dir']
)

```

### Batch Processing
Loop over subjects with joblib or similar—each instance is independent.

---

## 📁 Expected Project Layout
Your BIDS-like project must contain:
```
project_root/
├── derivatives/
│   └── ica/                 # ICA-cleaned FIF files
└── derivatives/lcmv/        # ← Must contain fsaverage (pre-downloaded!)
    └── fsaverage/
        ├── bem/
        │   ├── fsaverage-5120-5120-5120-bem-sol.fif
        │   └── fsaverage-head-dense.fif
        └── mri/T1.mgz
```

> Use `mne.datasets.fetch_fsaverage()` once to populate `derivatives/lcmv/fsaverage`.

---

## Dependencies
- MNE-Python ≥1.6
- Nilearn ≥0.10  
- Nibabel, Pandas, NumPy, SciPy

---

## Technical Details

**Source reconstruction**: LCMV beamforming was performed in fsaverage template. Coregistration employed three-stage procedure: fiducial-based alignment; Iterative Closest Point refinement using all 280 channels as head shape points (nasion weight 2.0) with outlier rejection (>5 mm MRI distance); final refinement (nasion weight 10.0, 20 iterations). Pipeline failed if mean coregistration error exceeded 5 mm (no identity transform fallback). Mean error was 4.86 mm. Five-millimeter isotropic volume grid with three-shell boundary element model was used. Single OAS-regularized covariance estimate from entire trial (noise covariance equal to data covariance). LCMV used max-power orientation selection, reduced rank, rank estimation accounting for average referencing, and Tikhonov regularization (λ = 0.01).


