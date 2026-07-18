# lcmv_xtra/coreg.py
"""Coregistration check — interactive 3D visualization using k3d (browser-rendered, no server GPU)."""
import json
import numpy as np
import k3d
import mne
from pathlib import Path
from typing import Optional


def plot_coregistration(
    fif_path: str,
    trans_file: str,
    head_surf_file: str,
    pipeline_metadata_file: str,
    add_brain: bool = False,
    lh_pial_file: Optional[str] = None,
    rh_pial_file: Optional[str] = None,
    lock_rotation: bool = True,
    show_grid: bool = False,
    show_axes: bool = True,
    head_opacity: float = 0.35,
    electrode_color: int = 0xe63946,
    electrode_size: float = 0.005,
    save_to: Optional[str] = None,
):
    """Interactive 3D coregistration quality check.

    Parameters
    ----------
    fif_path : str or Path
        Path to the cleaned .fif file.
    trans_file : str or Path
        Path to fsaverage-trans.fif.
    head_surf_file : str or Path
        Path to fsaverage-head-dense.fif.
    pipeline_metadata_file : str or Path
        Path to pipeline_metadata.json.
    add_brain : bool
        If True, overlay the pial brain surface.
    lh_pial_file, rh_pial_file : str or Path, optional
        Paths to lh.pial and rh.pial. Required if add_brain=True.
    lock_rotation : bool
        If True, lock Z-axis as up and prevent camera tilting (orbit mode).
    show_grid : bool
        If True, show the reference grid.
    show_axes : bool
        If True, show RAS coordinate axes.
    head_opacity : float
        Opacity of the head mesh (0-1).
    electrode_color : int
        Hex color for electrodes.
    electrode_size : float
        Point size for electrodes.
    save_to : str or Path, optional
        If provided, save an HTML snapshot.

    Returns
    -------
    k3d.Plot
    """
    fif_path = Path(fif_path)
    trans_file = Path(trans_file)
    head_surf_file = Path(head_surf_file)
    pipeline_metadata_file = Path(pipeline_metadata_file)

    # Load
    raw = mne.io.read_raw_fif(fif_path, preload=False)
    trans = mne.read_trans(str(trans_file))

    with open(pipeline_metadata_file) as f:
        meta = json.load(f)

    montage = raw.get_montage()
    if montage is None:
        raise ValueError("No montage found in the .fif file.")

    ch_pos = montage.get_positions()['ch_pos']
    electrode_coords = np.array(list(ch_pos.values()), dtype=np.float32)
    coreg_error = meta['coreg_mean_error_mm']
    subject_id = meta.get('subject_id', fif_path.stem)

    # Transform to MRI space
    head_surf = mne.read_bem_surfaces(str(head_surf_file))[0]
    head_to_mri = trans['trans']

    electrodes_homog = np.column_stack([electrode_coords, np.ones(len(electrode_coords))])
    electrodes_mri = (head_to_mri @ electrodes_homog.T).T[:, :3].astype(np.float32)

    # Fiducials — from montage positions (nasion/lpa/rpa stored directly)
    pos = montage.get_positions()
    fid_labels = {'nasion': 'Nasion', 'lpa': 'LPA', 'rpa': 'RPA'}
    fid_colors = {'Nasion': 0x00ff00, 'LPA': 0x0000ff, 'RPA': 0xff0000}
    fid_mri = {}
    for key, name in fid_labels.items():
        if key in pos and pos[key] is not None:
            coord = np.array([pos[key]], dtype=np.float32)
            coord_homog = np.column_stack([coord, np.ones(1)])
            fid_mri[name] = (head_to_mri @ coord_homog.T).T[:, :3].astype(np.float32)[0]

    # k3d plot
    plot_kwargs = dict(
        name=f"Coregistration — {subject_id} ({coreg_error:.2f} mm)",
        height=600,
        grid_visible=show_grid,
    )
    if lock_rotation:
        plot_kwargs['camera_mode'] = 'orbit'

    plot = k3d.plot(**plot_kwargs)

    # Head mesh
    head_verts = np.asarray(head_surf['rr'], dtype=np.float32)
    head_faces = np.asarray(head_surf['tris'], dtype=np.uint32)
    plot += k3d.mesh(head_verts, head_faces, color=0xd4c5b9, opacity=head_opacity, name='Head')

    # Optional brain
    if add_brain:
        if lh_pial_file is None or rh_pial_file is None:
            raise ValueError("lh_pial_file and rh_pial_file required when add_brain=True")
        for hemi, color, surf_path in [
            ('lh', 0x888888, Path(lh_pial_file)),
            ('rh', 0x999999, Path(rh_pial_file)),
        ]:
            if surf_path.exists():
                verts, faces = mne.read_surface(str(surf_path))
                verts = np.asarray(verts, dtype=np.float32) * 0.001
                faces = np.asarray(faces, dtype=np.uint32)
                plot += k3d.mesh(verts, faces, color=color, opacity=0.7,
                                 name=f'{hemi.upper()} Hemisphere')

    # Electrodes
    plot += k3d.points(
        electrodes_mri, color=electrode_color, point_size=electrode_size,
        name=f'Electrodes ({len(electrode_coords)})',
    )

    # Fiducials
    for name, coord in fid_mri.items():
        plot += k3d.points(
            np.array([coord], dtype=np.float32),
            color=fid_colors[name], point_size=0.012, name=name,
        )

    # Coordinate axes (RAS)
    if show_axes:
        origin = np.array([[0., 0., 0.]], dtype=np.float32)
        plot += k3d.line(np.vstack([origin, np.array([[0.05, 0., 0.]])]),
                         color=0xff0000, width=0.003, name='')
        plot += k3d.line(np.vstack([origin, np.array([[0., 0.05, 0.]])]),
                         color=0x00ff00, width=0.003, name='')
        plot += k3d.line(np.vstack([origin, np.array([[0., 0., 0.05]])]),
                         color=0x0000ff, width=0.003, name='')

    plot.camera = [0.0, -0.2, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0, 1.0]

    if save_to:
        with open(save_to, 'w') as f:
            f.write(plot.get_snapshot())

    return plot
