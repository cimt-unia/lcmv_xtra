# =============================================================================
# lcmv_xtra Visualization Examples
# =============================================================================

# ---------------------------------------------------------------------------
# 1. MNI Coordinates on fsaverage T1
# ---------------------------------------------------------------------------
from lcmv_xtra import plot_mni_orthoview

# Single coordinate — bare minimum
plot_mni_orthoview([-27, -18, 57])

# Single coordinate with a custom label
plot_mni_orthoview([-27, -18, 57], region_names=["Left M1"])

# Multiple coordinates with custom names and colors
plot_mni_orthoview(
    [[-27, -18, 57], [27, -18, 57], [-12, -15, -7]],
    region_names=["Left M1", "Right M1", "Left STN"],
    colors=["red", "blue", "green"],      # overrides auto-colormap
    figsize=(14, 5),                       # narrower figure
    marker_size=15,                         # bigger dots
    cmap='viridis',                         # colormap if colors not given
    show=True,                              # display immediately (default)
)

# ---------------------------------------------------------------------------
# 2. CIMT Atlas ROIs on MNI Template
# ---------------------------------------------------------------------------
from lcmv_xtra import plot_cimt_rois

# Single ROI — short name (e.g., "L_4_ROI")
plot_cimt_rois(7)

# Single ROI — full anatomical name
plot_cimt_rois(7, label_type="region_full_name")
# → "Primary Motor Cortex (Area 4)"

# Single ROI — hemisphere + full name
plot_cimt_rois(7, label_type="full")
# → "Left Primary Motor Cortex (Area 4)"

# Single ROI — custom title
plot_cimt_rois(7, title="My Motor Cortex")

# Single ROI — no title
plot_cimt_rois(7, title="")

# Multiple ROIs — auto-legend with color-matched ROI short names
plot_cimt_rois([7, 187, 8], label_type="full", title="CIMT Atlas")

# Multiple ROIs — change colormap
plot_cimt_rois([7, 187, 372], cmap='Set1', title="Motor Network")

# Multiple ROIs — save to file instead of showing
plot_cimt_rois([7, 187], label_type="full", save_to="motor_cortex.png")

# ---------------------------------------------------------------------------
# 3. PSD Comparison Between Two Conditions
# ---------------------------------------------------------------------------
from lcmv_xtra import plot_group_psd_comparison

EEG_DIR = "/mnt/movement/users/jaizor/xtra/derivatives/ocd/trials/dp02/eeg_tensor"

# Bare minimum — only required parameters
plot_group_psd_comparison(
    f"{EEG_DIR}/study_gain.npz",           # condition A
    f"{EEG_DIR}/study_loss.npz",           # condition B
    roi_indices=[0, 1],                     # which ROIs to plot (REQUIRED)
)

# Full example — all parameters
plot_group_psd_comparison(
    f"{EEG_DIR}/study_gain.npz",           # path to first .npz tensor
    f"{EEG_DIR}/study_loss.npz",           # path to second .npz tensor
    roi_indices=[0, 1],                     # ROI indices to plot
    roi_names=["R1", "R2"],                # display names (auto-detected if omitted)
    label_one="Gain",                       # legend label for condition A
    label_two="Loss",                       # legend label for condition B
    sfreq=250.0,                            # sampling freq (auto-detected if omitted)
    freq_max=50.0,                          # max frequency on x-axis
    window_sec=4.0,                         # Welch window length
    overlap=0.75,                           # overlap fraction
    ref_band=(1.0, 4.0),                   # delta band for alignment
    color_one='#1F77B4',                   # line color for condition A
    color_two='#D62728',                   # line color for condition B
    title_prefix="EEG",                    # prepended to each subplot title
    show=True,                              # display figures
    save_to="/tmp/psd",                    # save as /tmp/psd_ROI_0.png, etc.
)