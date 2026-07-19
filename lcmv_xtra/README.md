# Source Reconstruction

The source estimation stage transforms cleaned sensor-level EEG data into anatomically localized brain activity using a Linearly Constrained Minimum Variance (LCMV) beamformer. Implemented via the `lcmv_xtra` package, this pipeline executes five sequential computational stages specifically optimized to maximize spatial fidelity for deep subcortical and limbic structures.

<br>

## Framework

<br>

### 1. Sensor Space Preparation and Coordinate Normalization
Before source modeling, raw EEG data must be established in a standardized geometric frame. The pipeline loads ICA-cleaned FIF files and applies a custom BEL 280-channel montage derived from a bundled GPSC digitization file.

-   **Channel Renaming:** Raw EGI channel labels are mapped to standard MNE-compatible names (E1–E280). This translation is mandatory because forward solution routines require standardized nomenclature to correctly associate sensor positions with lead field columns; without it, topographic alignment fails silently.
-   **Coordinate Centering:** Sensor positions are normalized by subtracting the mean 3D position across all channels. Centering the head geometry at the origin reduces floating-point precision errors during subsequent high-dimensional matrix operations (forward solution, beamformer weights).
-   **Unit Conversion:** Coordinates are converted from millimeters to meters. MNE’s forward solver operates exclusively in SI units; failure to convert would produce lead fields scaled incorrectly by a factor of 1000, rendering source amplitudes physically meaningless.
-   **Fiducial Validation:** The pipeline explicitly verifies the presence of three anatomical landmarks (Nasion, Left Preauricular, Right Preauricular). These fiducials define the head coordinate frame origin and axes. Missing fiducials make coregistration mathematically undefined, triggering immediate failure to prevent misaligned reconstruction.
-   **Reference Restoration:** An average reference projection is applied if absent. Average referencing restores the zero-sum electrical potential constraint that may have been altered during ICA cleaning. This constraint is mathematically necessary for accurate EEG forward modeling because the potential field must sum to zero across all sensors.

<br>

### 2. Coregistration with Outlier-Robust ICP
Coregistration aligns the EEG sensor array to the fsaverage MRI template. This is the most critical step for subcortical signal fidelity, as even small misalignments can displace deep sources by centimeters.

-   **Fiducial-Based Initialization:** Alignment begins by rigidly matching the three EEG fiducials to their MRI counterparts. This provides a coarse initial transform; without it, Iterative Closest Point (ICP) optimization could converge to a local minimum with grossly incorrect alignment.
-   **Two-Stage ICP Refinement:** A staged ICP algorithm refines alignment using all available EEG sensor positions:
    -   *Stage 1:* Six iterations with moderate nasion weighting (2.0) establish a stable intermediate alignment without overfitting to outlier points.
    -   *Outlier Removal:* Sensor-to-MRI distances are computed. Points exceeding 5 mm are excluded as likely digitization errors or cap slippage artifacts. Including these would bias final alignment toward incorrect positions, systematically displacing deep source estimates.
    -   *Stage 2:* Twenty additional iterations with high nasion weighting (10.0) converge on the final transform using only inlier points. Elevated nasion weight anchors the anterior region where key subcortical structures of interest are located.
-   **Quality Control:** Mean, median, and maximum coregistration errors are logged in millimeters. A mean error exceeding 5 mm triggers a warning, as this threshold represents the approximate boundary for reliable subcortical localization with high-density EEG.

<br>

### 3. Volumetric Forward Solution Computation
The forward model predicts what each EEG sensor would record given a unit current dipole at every location in the brain. Unlike surface-constrained approaches, this pipeline uses a **volumetric source space** to explicitly capture deep generators.

-   **Source Space:** A 5 mm isotropic grid (`fsaverage-vol-5mm-src.fif`) spans the entire brain volume, including basal ganglia, thalamus, brainstem, and cerebellum. This ensures subcortical structures are modeled as independent dipoles rather than being erroneously projected onto the nearest cortical surface.
-   **Boundary Element Model (BEM):** A three-layer BEM solution (`fsaverage-5120-5120-5120-bem-sol.fif`) models conductivity boundaries between brain, skull, and scalp. This accounts for the smearing effect of the skull on EEG signals, which is especially critical for accurately localizing deep sources whose signals must traverse multiple tissue layers.
-   **Minimum Distance Constraint:** Sources within 5 mm of the inner skull boundary are excluded. Dipoles too close to conductivity discontinuities produce singular or near-singular lead field columns that destabilize beamformer inversion and create artifactual hotspots.

<br>

### 4. LCMV Beamformer Filter Construction
Spatial filters are computed to pass activity from each target voxel while suppressing interference from all other locations. Several design choices specifically address challenges of post-ICA EEG data and deep source recovery.

-   **Condition-Specific Covariance Estimation:** Data covariance is computed **separately for each experimental condition** using the Oracle Approximating Shrinkage (OAS) estimator with `rank=None`. This separation is theoretically critical because LCMV beamformers suffer from severe **signal cancellation** when sources are correlated: estimated source power decreases as a quadratic function of correlation ($\hat{\sigma}^2 = 1 - \rho^2$), and perfect synchrony yields zero output [1][2]. Mixing conditions introduces artificial cross-state correlations that the beamformer misinterprets as interference, causing catastrophic suppression of true neural signals—particularly for deep limbic structures whose weak signals are most vulnerable to cancellation [1]. Condition-specific estimation ensures the covariance matrix reflects only the intrinsic functional connectivity of each state, preserving signal integrity for valid cross-condition comparison [3][4]. OAS provides optimal regularization for this high-dimensional setting, and empirical rank estimation (`rank=None`) correctly handles reduced effective dimensionality from ICA component rejection [5][6].
-   **Regularization:** A Tikhonov regularization parameter of 0.05 stabilizes the inverse of the whitened covariance matrix. Without regularization, ill-conditioned covariances from aggressive ICA cleaning produce spatial filters with extreme weights that amplify noise in low-variance directions [7].
-   **Orientation Selection:** Filters use `pick_ori='max-power'`, selecting the dipole orientation at each voxel that maximizes output power. For volumetric grids where true orientation is unknown, this avoids signal cancellation from arbitrarily assigned orientations and eliminates the need for computationally expensive vector beamformers [7].
-   **Weight Normalization:** Unit-noise-gain normalization scales filter weights so that unit noise variance produces unit output variance at every voxel. This is the standard method in MNE-Python to address the well-known depth bias of beamformers, where superficial sources dominate simply because they generate larger sensor signals [7][8][9]. Deep subcortical sources thereby receive equal statistical weight relative to cortical sources without introducing arbitrary depth priors that risk false positives when forward model accuracy is limited [10][11].
-   **Rank Reduction:** `reduce_rank=True` removes zero-variance components from filter computation, preventing division-by-zero errors in degenerate subspaces without affecting the signal-bearing subspace [7].

<br>

### 5. Source Time Course Extraction and Atlas Projection
The final stage applies spatial filters to continuous EEG data and extracts region-specific time courses.

-   **Filter Application:** LCMV filters are applied to raw (continuous, non-epoched) EEG data, producing a source estimate matrix of shape `(n_sources × n_timepoints)` at 250/500 Hz. Operating on continuous data preserves full temporal dynamics needed for spectral analysis and avoids edge artifacts from epoch-based filtering.
-   **Atlas Extraction:** Source-space activations are extracted for all 448 ROIs in the CIMT Unified Atlas. Each ROI's time course is computed as the mean activation across all voxels falling within that region's volumetric mask, producing condition-specific tensors ready for spectral analysis.
-   **Metadata Preservation:** A JSON metadata file records all processing parameters (sampling rate, regularization value, coregistration error, source count, file paths). This enables exact reproducibility and allows downstream scripts to validate that source estimates were generated with consistent settings across subjects and conditions.

<br>

## References

1.  Kuznetsova, A., Nurislamova, Y., & Ossadtchi, A. (2021). Modified covariance beamformer for solving MEG inverse problem in the environment with correlated sources. *NeuroImage*, 228, 117677. [https://doi.org/10.1016/j.neuroimage.2020.117677](https://doi.org/10.1016/j.neuroimage.2020.117677) — Analytically proves quadratic signal cancellation in LCMV with correlated sources; validates condition-specific covariance to prevent cross-state interference.
2.  Sekihara, K., & Nagarajan, S. S. (2008). *Adaptive Spatial Filters for Electromagnetic Brain Imaging*. Springer. [https://doi.org/10.1007/978-3-540-79370-0](https://doi.org/10.1007/978-3-540-79370-0) — Comprehensive treatment of signal cancellation in correlated environments and LCMV constraints.
3.  Brookes, M. J., Zumer, J. M., Stevenson, C. M., Hale, J. R., Barnes, G. R., Vrba, J., & Morris, P. G. (2012). Optimising experimental design for MEG beamformer imaging. *NeuroImage*, 59(4), 3784–3796. — Validates using data covariance as noise estimate for continuous task paradigms.
4.  Hillebrand, A., & Barnes, G. R. (2005). Beamformer analysis of electromagnetic data. *NeuroImage*, 26(4), 1046–1058. — Establishes condition-specific covariance as best practice for adaptive beamforming.
5.  MNE-Python Contributors. (2024). mne.compute_raw_covariance. *MNE-Python Documentation*. [https://mne.tools/stable/generated/mne.compute_raw_covariance.html](https://mne.tools/stable/generated/mne.compute_raw_covariance.html) — Explains `rank=None` behavior for post-ICA data and automatic rank detection.
6.  Engemann, D. A., & Gramfort, A. (2015). Automated model selection in covariance estimation and spatial whitening of MEG and EEG signals. *NeuroImage*, 108, 328–342. — Validates automatic rank estimation after ICA/SSS cleaning.
7.  MNE-Python Contributors. (2024). Source reconstruction using an LCMV beamformer. *MNE-Python Tutorials*. [https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html](https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html) — Official tutorial recommending unit-noise-gain to "deal with depth bias" and documenting `pick_ori='max-power'` for volumetric grids.
8.  Van Veen, B. D., van Drongelen, W., Yuchtman, M., & Suzuki, A. (1997). Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. *IEEE Transactions on Biomedical Engineering*, 44(9), 867–880. [https://doi.org/10.1109/10.623056](https://doi.org/10.1109/10.623056) — Foundational derivation of LCMV beamformer and unit-noise-gain constraint for uniform noise floor across depths.
9.  Borgiotti, G. B., & Kaplan, J. D. (1979). Superresolution in array processing. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(5), 499–507. — Original formulation of unit-noise-gain minimum variance beamformer.
10. Buschermöhle, M., et al. (2024). Comparing the performance of beamformer algorithms. *NeuroImage*, 288, 120519. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10901088/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901088/) — Discusses noise levels and depth bias strategies across toolboxes; validates unit-noise-gain as accepted standard.
11. Hillebrand, A., & Barnes, G. R. (2002). A quantitative assessment of the sensitivity of whole-head MEG to activity in the adult human cortex. *NeuroImage*, 16(3), 638–650. [https://www.sciencedirect.com/science/article/pii/S105381190291102X](https://www.sciencedirect.com/science/article/pii/S105381190291102X) — Demonstrates risks of depth priors introducing false positives in beamformers when forward model is inaccurate.
