# lcmv_xtra/viz.py

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from typing import Optional, Tuple, Union, List
from nilearn import plotting, image


def plot_mni_orthoview(
    coordinates: list,
    region_names: Optional[list] = None,
    colors: Optional[list] = None,
    fs_dir: str = None,
    figsize: Tuple[int, int] = (18, 7),
    marker_size: int = 10,
    cmap: str = 'Purples_r',
    show: bool = True
) -> plt.Figure:
    """Plot MNI coordinates on fsaverage T1 MRI slices.

    Parameters
    ----------
    coordinates : list of [x, y, z] or list of lists
        MNI coordinates in millimeters.
    region_names : list of str, optional
        Labels for each coordinate. Defaults to Region_1, Region_2, ...
    colors : list, optional
        Colors for each marker. Defaults to colormap.
    fs_dir : str or Path, optional
        Path to fsaverage directory containing mri/T1.mgz.
        Defaults to the bundled fsaverage in the package.
    figsize : tuple
        Figure size (width, height).
    marker_size : int
        Size of the coordinate markers.
    cmap : str
        Matplotlib colormap for auto-generated colors.
    show : bool
        If True, display the figure immediately.

    Returns
    -------
    plt.Figure
    """
    import lcmv_xtra

    if fs_dir is None:
        fs_dir = Path(lcmv_xtra.__file__).parent.parent / 'data' / 'fsaverage'

    fs_dir = Path(fs_dir)
    coords_array = np.atleast_2d(coordinates).astype(float)
    n_coords = coords_array.shape[0]

    if region_names is None:
        region_names = [f"Region_{i+1}" for i in range(n_coords)]
    if colors is None:
        cmap_func = plt.colormaps[cmap]
        colors = [cmap_func(i / max(n_coords - 1, 1)) for i in range(n_coords)]

    t1_path = fs_dir / "mri" / "T1.mgz"
    if not t1_path.exists():
        raise FileNotFoundError(f"T1.mgz not found at {t1_path}")

    img = nib.load(str(t1_path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    affine = img.affine
    inv_affine = np.linalg.inv(affine)

    homog = np.column_stack([coords_array, np.ones(n_coords)])
    voxel_coords = (inv_affine @ homog.T).T[:, :3].round().astype(int)
    cx, cy, _ = voxel_coords.mean(axis=0).astype(int)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, dpi=120,
        gridspec_kw={'width_ratios': [1, 1.2]}
    )

    if n_coords > 1:
        legend_items = [
            plt.Line2D(
                [0], [0], marker='o', color='w', label=name,
                markerfacecolor=color, markersize=10, markeredgewidth=2.5
            )
            for name, color in zip(region_names, colors)
        ]
        fig.legend(
            handles=legend_items, loc='center right',
            bbox_to_anchor=(1.0, 0.5), frameon=True,
            framealpha=0.9, fontsize=11, borderaxespad=0.5
        )

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
            ax.set_title(
                f"{view_name} | Slice = {center}", fontsize=12, fontweight='bold'
            )
        else:
            ax.text(
                0.5, 0.5, "Out of Range", ha="center", color="red",
                transform=ax.transAxes
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for voxel, color in zip(voxel_coords, colors):
            plot_x, plot_y = coord_func(voxel)
            ax.plot(
                plot_x, plot_y, 'o', color=color, ms=marker_size,
                mfc='none', mew=2.5
            )
            ax.axvline(plot_x, color=color, ls='--', alpha=0.5, lw=1)
            ax.axhline(plot_y, color=color, ls='--', alpha=0.5, lw=1)

    title = (
        f"Coordinate: {region_names[0]}"
        if n_coords == 1
        else f"Brain Locations: {n_coords} Region(s)"
    )
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    fig.subplots_adjust(
        right=0.85 if n_coords > 1 else 0.95,
        wspace=0.35, top=0.95, left=0.08
    )

    if show:
        plt.show()
    return fig


def plot_cimt_rois(
    indices: Union[int, List[int]],
    title: Optional[str] = None,
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
        For multiple ROIs, each gets a different color from the colormap.
    title : str, optional
        Plot title. Auto-generated if None.
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

    if isinstance(indices, int):
        indices = [indices]

    atlas_path = (
        Path(lcmv_xtra.__file__).parent
        / 'data' / 'cimt_atlas' / 'CIMT_448ROIs_atlas.nii.gz'
    )
    if not atlas_path.exists():
        raise FileNotFoundError(f"CIMT atlas not found at {atlas_path}")

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(np.int32)

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

    if title is None:
        n = len(indices)
        title = f"CIMT: {n} ROI{'s' if n > 1 else ''}"

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
        output_file=save_to,
    )

    if show and save_to is None:
        plotting.show()
