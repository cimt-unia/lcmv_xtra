# lcmv_xtra/viz.py

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import mne
import scipy.signal as signal
from pathlib import Path
from typing import Tuple, Optional
from scipy.integrate import trapezoid


# Global configuration
SFREQ = 500.0
PSD_WINDOW_SEC = 2.0

# Standard neurophysiological frequency bands (Hz)
FREQ_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Low_Beta': (12, 20),
    'High_Beta': (20, 30),
    'Low_Gamma': (30, 50),
    'High_Gamma': (50, 100)
}


def plot_mni_orthoview(
    coordinates: list,
    region_names: Optional[list] = None,
    colors: Optional[list] = None,
    base_dir: str = "/mnt/movement/users/jaizor/xtra/derivatives/lcmv",
    figsize: Tuple[int, int] = (18, 7),
    marker_size: int = 10,
    cmap: str = 'Purples_r',
    show: bool = True
) -> plt.Figure:
    """Plot coordinates on fsaverage T1 MRI (sagittal + coronal views)."""
    coords_array = np.atleast_2d(coordinates).astype(float)
    n_coords = coords_array.shape[0]

    if region_names is None:
        region_names = [f"Region_{i+1}" for i in range(n_coords)]
    if colors is None:
        cmap_func = plt.colormaps[cmap]
        colors = [cmap_func(i % cmap_func.N) for i in range(n_coords)]

    t1_path = Path(base_dir) / "fsaverage/mri/T1.mgz"
    if not t1_path.exists():
        raise FileNotFoundError(f"T1.mgz not found at {t1_path}")

    img = nib.load(str(t1_path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    inv_affine = np.linalg.inv(img.affine)

    homog = np.column_stack([coords_array, np.ones(n_coords)])
    voxel_coords = (inv_affine @ homog.T).T[:, :3].round().astype(int)
    cx, cy, _ = voxel_coords.mean(axis=0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120, 
                             gridspec_kw={'width_ratios': [1, 1.2]})

    if n_coords > 1:
        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', label=name,
                       markerfacecolor=color, markersize=10, markeredgewidth=2.5)
            for name, color in zip(region_names, colors)
        ]
        fig.legend(handles=legend_items, loc='center right', 
                   bbox_to_anchor=(1.0, 0.5), frameon=True, 
                   framealpha=0.9, fontsize=11, borderaxespad=0.5)

    views = [
        (cx, data[cx, :, :], "Sagittal", "Y (P ← → A)", "Z (I ← → S)", 
         lambda v: (v[1], v[2])),
        (cy, data[:, cy, :], "Coronal", "X (L ← → R)", "Z (I ← → S)", 
         lambda v: (v[0], v[2]))
    ]

    for ax_idx, (center, slice_data, view_name, xlabel, ylabel, coord_func) in enumerate(views):
        ax = axes[ax_idx]
        if 0 <= center < slice_data.shape[0]:
            ax.imshow(slice_data.T, cmap="gray", origin="lower")
            ax.set_title(f"{view_name} | Slice = {center}", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "Out of Range", ha="center", color="red", 
                    transform=ax.transAxes)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for voxel, color in zip(voxel_coords, colors):
            plot_x, plot_y = coord_func(voxel)
            ax.plot(plot_x, plot_y, 'o', color=color, ms=marker_size, 
                    mfc='none', mew=2.5)
            ax.axvline(plot_x, color=color, ls='--', alpha=0.5, lw=1)
            ax.axhline(plot_y, color=color, ls='--', alpha=0.5, lw=1)

    title = (f"Coordinate: {region_names[0]}" if n_coords == 1 
             else f"Brain Locations: {n_coords} Region(s)")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    fig.subplots_adjust(right=0.85 if n_coords > 1 else 0.95, 
                        wspace=0.35, top=0.95, left=0.08)

    if show:
        plt.show()
    return fig


def _compute_band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """Compute integrated power in standard neurophysiological bands."""
    band_powers = {}
    df = freqs[1] - freqs[0]
    for band, (fmin, fmax) in FREQ_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            band_powers[band] = trapezoid(psd[mask], dx=df)
        else:
            band_powers[band] = 0.0
    return band_powers


def compute_psd(
    time_series: np.ndarray,
    sfreq: float = SFREQ,
    method: str = 'welch',
    fmin: float = 1.0,
    fmax: float = 100.0,
    window_sec: float = PSD_WINDOW_SEC
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute PSD using Welch or Multitaper method."""
    ts = np.real(time_series).astype(np.float64)
    window_size = int(window_sec * sfreq)
    if len(ts) < window_size:
        raise ValueError("Time series too short for PSD estimation.")

    # High-pass filter (0.5 Hz)
    nyq = sfreq * 0.5
    b, a = signal.butter(4, 0.5 / nyq, btype='high')
    filtered = signal.filtfilt(b, a, ts)

    if method == 'welch':
        freqs, psd = signal.welch(
            filtered, fs=sfreq, window='hann', nperseg=window_size,
            noverlap=window_size // 2, detrend='constant'
        )
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs, psd = freqs[mask], psd[mask]

    elif method == 'multitaper':
        try:
            from mne.time_frequency import psd_array_multitaper
            psd, freqs = psd_array_multitaper(
                x=filtered, sfreq=sfreq, fmin=fmin, fmax=fmax,
                bandwidth=2.0, adaptive=False, normalization='length',
                low_bias=True, verbose=False
            )
        except Exception as e:
            print(f"Multitaper failed ({e}), falling back to Welch.")
            return compute_psd(time_series, sfreq, 'welch', fmin, fmax, window_sec)
    else:
        raise ValueError("method must be 'welch' or 'multitaper'")

    band_powers = _compute_band_powers(freqs, psd)
    return freqs.astype(np.float32), psd.astype(np.float32), band_powers


def visualize_source_at_coordinate(
    stc_path: str,
    mni_coord: list,
    roi_name: str = "Custom ROI",
    base_dir: str = "/mnt/movement/users/jaizor/xtra/derivatives/_fs",
    sfreq: float = SFREQ,
    psd_method: str = 'welch',
    psd_color: str = "#4A3387",
    band_cmap: str = "Blues"
):
    """
    Visualize PSD and brain location for a given MNI coordinate.
    
    This is your QC function - use it BEFORE batch processing to verify
    that your MNI coordinates target the correct brain regions.
    
    Parameters
    ----------
    stc_path : str
        Path to source estimate file (.h5 or .stc)
    mni_coord : list
        Target MNI coordinate [x, y, z] in mm
    roi_name : str
        Label for the region
    base_dir : str
        Directory containing fsaverage/ and source space files
    sfreq : float
        Sampling frequency (Hz)
    psd_method : str
        'welch' or 'multitaper'
    psd_color : str
        Color for the main PSD line
    band_cmap : str
        Colormap for frequency band shading
    """
    # Load STC and source space
    stc = mne.read_source_estimate(stc_path)
    src_file = Path(base_dir) / "fsaverage-vol-5mm-src.fif"
    src = mne.read_source_spaces(str(src_file))

    # Get active source coordinates (MNI mm) - using proper MNI transform
    from nilearn import image
    
    active_vertices = stc.vertices[0]
    src_rr = src[0]['rr'][active_vertices] * 1000  # m → mm
    
    try:
        trans = src[0]['mri_ras_t']['trans']
    except KeyError:
        raise ValueError("Source space missing 'mri_ras_t' transform.")
    
    mni_coords = image.coord_transform(
        src_rr[:, 0], src_rr[:, 1], src_rr[:, 2], trans
    )
    active_coords_mm = np.array(mni_coords).T

    if stc.data.shape[0] != len(active_coords_mm):
        raise RuntimeError("Active source count mismatch")

    # Find closest active source
    target = np.array(mni_coord)
    distances = np.linalg.norm(active_coords_mm - target, axis=1)
    best_idx_in_active = np.argmin(distances)
    actual_coord = active_coords_mm[best_idx_in_active]

    print(f"Requested MNI: {mni_coord}")
    print(f"Closest active source: [{actual_coord[0]:.1f}, "
          f"{actual_coord[1]:.1f}, {actual_coord[2]:.1f}] "
          f"(dist: {distances[best_idx_in_active]:.1f} mm)")

    # Plot orthoview
    plot_mni_orthoview(
        coordinates=[mni_coord], 
        region_names=[roi_name], 
        base_dir=base_dir
    )

    # Extract time series and compute PSD
    ts = stc.data[best_idx_in_active, :]
    freqs, psd, band_powers = compute_psd(
        ts, sfreq=sfreq, method=psd_method, fmin=1.0, fmax=100.0
    )

    # Plot PSD
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, psd, color=psd_color, linewidth=2.2)

    # Band shading
    cmap = plt.colormaps.get_cmap(band_cmap)
    n_bands = len(FREQ_BANDS)
    
    for i, (band, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        if fmax < freqs[0] or fmin > freqs[-1]:
            continue
        color_intensity = 0.3 + (i / max(n_bands - 1, 1)) * 0.4
        band_color = cmap(color_intensity)
        band_low = max(fmin, freqs[0])
        band_high = min(fmax, freqs[-1])
        ax.axvspan(band_low, band_high, color=band_color, alpha=0.10, zorder=0)

    # Band boundaries
    all_boundaries = set()
    for fmin, fmax in FREQ_BANDS.values():
        if freqs[0] <= fmin <= freqs[-1]:
            all_boundaries.add(fmin)
        if freqs[0] <= fmax <= freqs[-1]:
            all_boundaries.add(fmax)
    
    for boundary in sorted(all_boundaries):
        ax.axvline(boundary, color='gray', linestyle=':', alpha=0.4, 
                   linewidth=0.6, zorder=1)

    # Band labels
    for band, (low, high) in FREQ_BANDS.items():
        if high < freqs[0] or low > freqs[-1]:
            continue
        center_x = (max(low, freqs[0]) + min(high, freqs[-1])) / 2
        ax.text(center_x, 1.02, band.replace('_', ' '), 
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, 
                color='dimgray', alpha=0.85, fontweight='regular')

    # Formatting
    y_label = f"PSD (log₁₀) - {psd_method.capitalize()}"
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(1, 100)
    ax.grid(True, which='both', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.suptitle(f"{roi_name} — Power Spectral Density", 
                 fontsize=14, fontweight='bold', y=0.98)

    plt.show()
