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
    cmap: str = 'Set1',
    show: bool = True
) -> plt.Figure:
    """Plot over fsaverage T1 from base_dir/fsaverage/mri/T1.mgz (1 mm isotropic)."""
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

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120, gridspec_kw={'width_ratios': [1, 1.2]})

    if n_coords > 1:
        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', label=name,
                       markerfacecolor=color, markersize=10, markeredgewidth=2.5)
            for name, color in zip(region_names, colors)
        ]
        fig.legend(handles=legend_items, loc='center right', bbox_to_anchor=(1.0, 0.5),
                   frameon=True, framealpha=0.9, fontsize=11, borderaxespad=0.5)

    views = [
        (cx, data[cx, :, :], "Sagittal", "Y (P ← → A)", "Z (I ← → S)", lambda v: (v[1], v[2])),
        (cy, data[:, cy, :], "Coronal", "X (L ← → R)", "Z (I ← → S)", lambda v: (v[0], v[2]))
    ]

    for ax_idx, (center, slice_data, view_name, xlabel, ylabel, coord_func) in enumerate(views):
        ax = axes[ax_idx]
        if 0 <= center < slice_data.shape[0]:
            ax.imshow(slice_data.T, cmap="gray", origin="lower")
            ax.set_title(f"{view_name} | Slice = {center}", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "Out of Range", ha="center", color="red", transform=ax.transAxes)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for voxel, color in zip(voxel_coords, colors):
            plot_x, plot_y = coord_func(voxel)
            ax.plot(plot_x, plot_y, 'o', color=color, ms=marker_size, mfc='none', mew=2.5)
            ax.axvline(plot_x, color=color, ls='--', alpha=0.5, lw=1)
            ax.axhline(plot_y, color=color, ls='--', alpha=0.5, lw=1)

    title = f"Coordinate: {region_names[0]}" if n_coords == 1 else f"Brain Locations: {n_coords} Region(s)"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    fig.subplots_adjust(right=0.85 if n_coords > 1 else 0.95, wspace=0.35, top=0.85, left=0.08)

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
    """
    Compute PSD using Welch or Multitaper.
    """
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
    base_dir: str = "/mnt/movement/users/jaizor/xtra/derivatives/lcmv",
    sfreq: float = SFREQ,
    psd_method: str = 'welch',
    band_cmap: str = 'magma_r',
    psd_color: str = '#2E3440'  # Dark gray from Nord palette
):
    """
    Visualize full-spectrum PSD at a given MNI coordinate.
    
    Parameters:
    - stc_path: Path to source estimate file (.h5 or .stc)
    - mni_coord: Target MNI coordinate [x, y, z] in mm
    - roi_name: Label for the region
    - base_dir: Directory containing fsaverage/ and source space files
    - sfreq: Sampling frequency (Hz)
    - psd_method: 'welch' or 'multitaper'
    - band_cmap: Colormap for frequency band highlighting (default: 'magma_r')
    - psd_color: Color for the main PSD line (default: dark gray)
    """
    # Load STC and source space
    stc = mne.read_source_estimate(stc_path)
    src_file = Path(base_dir) / "fsaverage-vol-5mm-src.fif"
    src = mne.read_source_spaces(str(src_file))

    # Get active source coordinates (MNI mm)
    active_vertices = stc.vertices[0]
    active_coords_m = src[0]['rr'][active_vertices]
    active_coords_mm = active_coords_m * 1000  # m → mm

    if stc.data.shape[0] != len(active_coords_mm):
        raise RuntimeError("Active source count mismatch")

    # Find closest active source
    target = np.array(mni_coord)
    distances = np.linalg.norm(active_coords_mm - target, axis=1)
    best_idx_in_active = np.argmin(distances)
    actual_coord = active_coords_mm[best_idx_in_active]

    print(f"Requested: {mni_coord}")
    print(f"Closest active source: [{actual_coord[0]:.1f}, {actual_coord[1]:.1f}, {actual_coord[2]:.1f}] "
          f"(dist: {distances[best_idx_in_active]:.1f} mm)")

    # Plot orthoview
    plot_mni_orthoview(coordinates=[mni_coord], region_names=[roi_name], base_dir=base_dir)

    # Extract time series
    ts = stc.data[best_idx_in_active, :]

    # Compute PSD
    freqs, psd, band_powers = compute_psd(ts, sfreq=sfreq, method=psd_method, fmin=1.0, fmax=100.0)

    # Plot full-spectrum PSD with improved aesthetics
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Main PSD line with improved styling
    ax.plot(freqs, psd, color=psd_color, linewidth=2.5, alpha=0.9, label='PSD')
    
    # Enhanced band coloring with better aesthetic mapping
    bands_list = list(FREQ_BANDS.keys())
    n_bands = len(bands_list)
    
    try:
        cmap_func = plt.colormaps[band_cmap]
    except KeyError:
        print(f"Warning: colormap '{band_cmap}' not found. Using 'magma_r'")
        cmap_func = plt.colormaps['magma_r']
    
    # Create harmonious band colors
    if n_bands == 1:
        band_colors = {bands_list[0]: cmap_func(0.7)}
    else:
        # Use perceptually uniform spacing across the colormap
        band_colors = {
            band: cmap_func(0.2 + 0.7 * i / (n_bands - 1))
            for i, band in enumerate(bands_list)
        }

    # Shade each band with improved styling
    for band, (fmin, fmax) in FREQ_BANDS.items():
        alpha_val = 0.18 if band in ['Alpha', 'Beta'] else 0.12  # Emphasize key bands
        ax.axvspan(fmin, fmax, color=band_colors[band], alpha=alpha_val,
                   label=band.replace('_', ' '), zorder=0)

    # Enhanced styling
    ax.set_title(f"{roi_name} — Power Spectral Density ({psd_method.capitalize()})", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight='medium')
    ax.set_ylabel("Power Spectral Density", fontsize=12, fontweight='medium')
    ax.set_yscale('log')
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    
    # Improved legend positioning
    if n_bands <= 4:
        legend_loc = 'upper right'
        ncol = 1
    else:
        legend_loc = 'upper center'
        ncol = 2 if n_bands <= 8 else 3
    
    ax.legend(loc=legend_loc, fontsize=10, ncol=ncol, 
              framealpha=0.9, fancybox=True, shadow=False,
              title="Frequency Bands", title_fontsize=11)
    
    plt.tight_layout()
    plt.show()

    # Print integrated band powers with better formatting
    print("\n📊 Band Powers (integrated):")
    for band, power in band_powers.items():
        display_name = band.replace('_', ' ')
        print(f"  {display_name:<12}: {power:.2e}")

'''
# EXAMPLE USAGE

from lcmv_xtra.viz import visualize_source_at_coordinate

visualize_source_at_coordinate(
    stc_path="/path/lcmv/sub-001/source_estimate_LCMV.h5",
    mni_coord=[-42, -18, 56],
    roi_name="Left M1",
    base_dir="/mnt/movement/users/jaizor/xtra/derivatives/_fs",
    psd_method='welch'  # or 'multitaper'
)

'''
