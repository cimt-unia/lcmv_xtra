# lcmv_xtra/viz.py

import logging
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from scipy import signal
from typing import Optional, Tuple, Union, List
from nilearn import plotting, image

def plot_mni_orthoview(
    coordinates: list,
    region_names: Optional[list] = None,
    colors: Optional[list] = None,
    figsize: Tuple[int, int] = (18, 7),
    marker_size: int = 10,
    cmap: str = 'Purples_r',
    show: bool = True
) -> plt.Figure:
    import lcmv_xtra

    t1_path = Path(lcmv_xtra.__file__).parent / 'data' / 'fsavg' / 'T1.mgz'

    coords_array = np.atleast_2d(coordinates).astype(float)
    n_coords = coords_array.shape[0]

    if region_names is None:
        region_names = [f"Region_{i+1}" for i in range(n_coords)]
    if colors is None:
        cmap_func = plt.colormaps[cmap]
        colors = [cmap_func(i / max(n_coords - 1, 1)) for i in range(n_coords)]

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
         lambda v: (v[0], v[2])),
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

    if show:
        plt.show()
    plt.close(fig)
    return fig

def plot_cimt_rois(
    indices: Union[int, List[int]],
    title: Optional[str] = None,
    label_type: str = "roi_name",
    cmap: str = 'coolwarm',
    alpha: float = 0.7,
    show: bool = True,
    save_to: Optional[str] = None,
) -> None:
    """Plot CIMT atlas ROIs on MNI template using ortho view.

    Parameters
    ----------
    indices : int or list of int
        CIMT ROI indices (0-447). Maps to NIfTI labels 1-448.
    title : str, optional
        Plot title. Auto-generated from label_type if None.
        Use '' for no title.
    label_type : str
        Column from cimt_atlas_labels.csv for auto-title:
        'roi_name' (e.g., 'L_4_ROI'),
        'region_full_name' (e.g., 'Primary Motor Cortex (Area 4)'),
        or 'full' for 'hemisphere region_full_name'.
    cmap : str
        Matplotlib colormap for ROIs.
    alpha : float
        Opacity of the ROI overlay (0-1).
    show : bool
        If True, display the figure immediately.
    save_to : str or Path, optional
        If provided, save the figure to this path.
    """
    import lcmv_xtra
    import pandas as pd

    if isinstance(indices, int):
        indices = [indices]

    atlas_path = (
        Path(lcmv_xtra.__file__).parent
        / 'data' / 'cimt_atlas' / 'CIMT_448ROIs_atlas.nii.gz'
    )
    labels_path = (
        Path(lcmv_xtra.__file__).parent
        / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'
    )

    if not atlas_path.exists():
        raise FileNotFoundError(f"CIMT atlas not found at {atlas_path}")

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    roi_df = pd.read_csv(labels_path)

    # Build mask with sequential values for colormap
    mask = np.zeros(atlas_data.shape, dtype=np.int32)
    for i, idx in enumerate(indices):
        mask[atlas_data == idx + 1] = i + 1

    mask_img = image.new_img_like(atlas_img, mask)

    # Center view on the midpoint of all selected ROIs
    all_voxels = np.argwhere(mask > 0)
    if len(all_voxels) == 0:
        raise ValueError("None of the requested labels found in the atlas.")
    com = nib.affines.apply_affine(atlas_img.affine, all_voxels.mean(axis=0))
    cut_coords = tuple(com.round().astype(int))

    # Build title
    if title is None:
        if len(indices) == 1:
            row = roi_df.iloc[indices[0]]
            if label_type == "full":
                title = f"{row['hemisphere']} {row['region_full_name']}"
            else:
                title = str(row[label_type])
        else:
            title = f"CIMT: {len(indices)} ROIs"
    elif title == "":
        title = None

    plotting.plot_roi(
        mask_img,
        title=title,
        cut_coords=cut_coords,
        display_mode='ortho',
        cmap=cmap,
        alpha=alpha,
        dim=-0.5,
        black_bg=False,
        draw_cross=True,
        radiological=False,
        colorbar=False,
        output_file=save_to,
    )

    # Add ROI name labels for multi-ROI plots
    if len(indices) > 1:
        names = []
        for idx in indices:
            row = roi_df.iloc[idx]
            if label_type == "full":
                names.append(f"{row['hemisphere']} {row['region_full_name']}")
            else:
                names.append(str(row[label_type]))
        label_text = "\n".join(names)
        plt.gcf().text(
            0.82, 0.5, label_text,
            transform=plt.gcf().transFigure,
            fontsize=9, verticalalignment='center',
            fontfamily='monospace',
        )

    if show and save_to is None:
        plotting.show()

def plot_group_psd_comparison(
    condition_one_path: str,
    condition_two_path: str,
    label_one: str = "Condition 1",
    label_two: str = "Condition 2",
    roi_indices: Optional[List[int]] = None,
    roi_names: Optional[List[str]] = None,
    sfreq: Optional[float] = None,
    freq_max: float = 50.0,
    window_sec: float = 4.0,
    overlap: float = 0.75,
    ref_band: Tuple[float, float] = (1.0, 4.0),
    color_one: str = '#1F77B4',
    color_two: str = '#D62728',
    title_prefix: str = "",
    show: bool = True,
    save_to: Optional[str] = None,
) -> List[plt.Figure]:
    """Plot PSD comparison between two conditions from study tensors.

    Parameters
    ----------
    condition_one_path, condition_two_path : str or Path
        Paths to .npz files from assemble_tensor() or assemble_custom_tensor().
        Each must contain 'data' (subjects, ROIs, time) and 'sfreq'.
    label_one, label_two : str
        Display names for the two conditions.
    roi_indices : list of int, optional
        Which ROI indices to plot. If None, plots all ROIs.
    roi_names : list of str, optional
        Names for each ROI index. If None, uses generic names.
        For CIMT tensors, this can be loaded from the bundled CSV.
    sfreq : float, optional
        Sampling frequency. Auto-detected from .npz if None.
    freq_max : float
        Maximum frequency to display (Hz).
    window_sec : float
        Welch window length in seconds.
    overlap : float
        Fraction of window overlap for Welch.
    ref_band : tuple
        (fmin, fmax) for delta-alignment reference band.
    color_one, color_two : str
        Line colors for the two conditions.
    title_prefix : str
        Prepended to each subplot title.
    show : bool
        If True, display figures.
    save_to : str or Path, optional
        If provided, save figures with this prefix (e.g., 'psd' → 'psd_ROI_0.png').

    Returns
    -------
    list of plt.Figure
    """
    import logging
    logger = logging.getLogger(__name__)

    cond_one = np.load(condition_one_path, allow_pickle=True)
    cond_two = np.load(condition_two_path, allow_pickle=True)

    data_one = cond_one['data']   # (subjects, ROIs, time)
    data_two = cond_two['data']

    if sfreq is None:
        sfreq = float(cond_one.get('sfreq', 250.0))
    sfreq = float(sfreq)

    n_rois = data_one.shape[1]
    if data_two.shape[1] != n_rois:
        raise ValueError(
            f"ROI count mismatch: {data_one.shape[1]} vs {data_two.shape[1]}"
        )

    # Determine which ROIs to plot
    if roi_indices is None:
        roi_indices = list(range(n_rois))
    if roi_names is None:
        # Try to get from .npz (custom tensor) or use generic
        if 'roi_names' in cond_one:
            all_names = list(cond_one['roi_names'])
        else:
            all_names = [f"ROI_{i}" for i in range(n_rois)]
        roi_names = [all_names[i] for i in roi_indices]

    # Average across subjects
    mean_one = np.nanmean(data_one, axis=0)  # (ROIs, time)
    mean_two = np.nanmean(data_two, axis=0)

    # PSD bands for shading
    bands = [
        (1, 4, 'Delta', '#90B3F9'),
        (4, 8, 'Theta', '#FFF9B2'),
        (8, 13, 'Alpha', '#AAFCD2'),
        (13, 20, 'Low Beta', '#97C2F9'),
        (20, 30, 'High Beta', '#90BEF5'),
    ]

    nperseg = min(int(sfreq * window_sec), data_one.shape[2])
    noverlap = int(nperseg * overlap)
    eps = 1e-15

    figures = []

    for i, roi_idx in enumerate(roi_indices):
        sig_one = mean_one[roi_idx]
        sig_two = mean_two[roi_idx]

        # Compute PSDs
        freqs, psd_one_lin = signal.welch(
            sig_one, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window='hann'
        )
        _, psd_two_lin = signal.welch(
            sig_two, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window='hann'
        )

        # Delta alignment
        mask_ref = (freqs >= ref_band[0]) & (freqs <= ref_band[1])
        ref_one = psd_one_lin[mask_ref].mean()
        ref_two = psd_two_lin[mask_ref].mean()
        offset = 10 * np.log10((ref_one + eps) / (ref_two + eps))

        psd_one_db = 10 * np.log10(psd_one_lin + eps)
        psd_two_db = 10 * np.log10(psd_two_lin + eps) + offset

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        mask_freq = freqs <= freq_max

        for f_lo, f_hi, name, color in bands:
            if f_hi <= freq_max:
                ax.axvspan(f_lo, f_hi, facecolor=color, alpha=0.08, edgecolor='none')
                ax.annotate(
                    name, xy=((f_lo + f_hi) / 2, 0.97),
                    xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=8, color='#1E3A5F',
                    fontweight='bold', alpha=0.8,
                )

        ax.plot(freqs[mask_freq], psd_one_db[mask_freq], color=color_one, lw=2, label=label_one)
        ax.plot(freqs[mask_freq], psd_two_db[mask_freq], color=color_two, lw=2, label=label_two)

        ax.set_xlim(1, freq_max)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB)')
        title = f"{title_prefix + ' — ' if title_prefix else ''}{roi_names[i]}"
        ax.set_title(title, fontsize=12, color='#1E3A5F', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25, ls='--')
        fig.tight_layout()

        if save_to:
            save_path = Path(save_to)
            fig.savefig(
                save_path.parent / f"{save_path.stem}_ROI_{roi_idx}.png",
                dpi=150, bbox_inches='tight'
            )

        figures.append(fig)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures
