# lcmv_xtra/coreg.py
"""Coregistration check — interactive 3D visualization using k3d (browser-rendered, no server GPU)."""

import json
import logging
from pathlib import Path
from typing import Optional

import k3d
import mne
import numpy as np

logger = logging.getLogger(__name__)

STANDALONE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/k3d@{k3d_version}/dist/k3d.min.js"></script>
</head>
<body>
    <div id="plot" style="width:100%; height:{height}px;"></div>
    <script>
        var plotJson = {plot_json};
        K3D.load(plotJson, document.getElementById("plot"));
    </script>
</body>
</html>"""

DEFAULT_PLOT_HEIGHT = 600
ELECTRODE_POINT_SIZE = 0.005
FIDUCIAL_POINT_SIZE = 0.012
HEAD_MESH_OPACITY = 0.35
BRAIN_MESH_OPACITY = 0.7
AXIS_LINE_WIDTH = 0.003
AXIS_LENGTH = 0.05
SURFACE_SCALE_MM_TO_M = 0.001


def _load_coreg_data(
    fif_path: Path,
    trans_file: Path,
    head_surf_file: Path,
    pipeline_metadata_file: Path,
) -> dict:
    """Load and transform coregistration data into MRI space."""
    raw = mne.io.read_raw_fif(fif_path, preload=False)
    trans = mne.read_trans(str(trans_file))

    with open(pipeline_metadata_file) as f:
        meta = json.load(f)

    montage = raw.get_montage()
    if montage is None:
        raise ValueError("No montage found in the .fif file.")

    ch_pos = montage.get_positions()["ch_pos"]
    electrode_coords = np.array(list(ch_pos.values()), dtype=np.float32)
    head_to_mri = trans["trans"]

    electrodes_homog = np.column_stack([electrode_coords, np.ones(len(electrode_coords))])
    electrodes_mri = (head_to_mri @ electrodes_homog.T).T[:, :3].astype(np.float32)

    fid_labels = {"FidNz": "Nasion", "FidT9": "LPA", "FidT10": "RPA"}
    fid_mri = {}
    for key, name in fid_labels.items():
        if key in ch_pos:
            coord = np.array([ch_pos[key]], dtype=np.float32)
            coord_homog = np.column_stack([coord, np.ones(1)])
            fid_mri[name] = (head_to_mri @ coord_homog.T).T[:, :3].astype(np.float32)[0]

    head_surf = mne.read_bem_surfaces(str(head_surf_file))[0]

    return {
        "electrodes_mri": electrodes_mri,
        "fid_mri": fid_mri,
        "head_surf": head_surf,
        "coreg_error": meta["coreg_mean_error_mm"],
        "subject_id": meta.get("subject_id", fif_path.stem),
    }


def _build_plot(data: dict, add_brain: bool, lh_pial_file: Optional[Path], rh_pial_file: Optional[Path]) -> k3d.Plot:
    """Construct the k3d plot object from pre-loaded data."""
    plot = k3d.plot(
        name=f"Coregistration — {data['subject_id']} ({data['coreg_error']:.2f} mm)",
        height=DEFAULT_PLOT_HEIGHT,
    )

    head_verts = np.asarray(data["head_surf"]["rr"], dtype=np.float32)
    head_faces = np.asarray(data["head_surf"]["tris"], dtype=np.uint32)
    plot += k3d.mesh(head_verts, head_faces, color=0xD4C5B9, opacity=HEAD_MESH_OPACITY, name="Head")

    if add_brain:
        if lh_pial_file is None or rh_pial_file is None:
            raise ValueError("lh_pial_file and rh_pial_file required when add_brain=True")
        hemi_configs = [
            ("lh", 0x888888, lh_pial_file),
            ("rh", 0x999999, rh_pial_file),
        ]
        for hemi, color, surf_path in hemi_configs:
            if surf_path.exists():
                verts, faces = mne.read_surface(str(surf_path))
                verts = np.asarray(verts, dtype=np.float32) * SURFACE_SCALE_MM_TO_M
                faces = np.asarray(faces, dtype=np.uint32)
                plot += k3d.mesh(verts, faces, color=color, opacity=BRAIN_MESH_OPACITY, name=f"{hemi.upper()} Hemisphere")

    plot += k3d.points(
        data["electrodes_mri"],
        color=0xE63946,
        point_size=ELECTRODE_POINT_SIZE,
        name=f"Electrodes ({len(data['electrodes_mri'])})",
    )

    fid_colors = {"Nasion": 0x2A9D8F, "LPA": 0x264653, "RPA": 0xE9C46A}
    for name, coord in data["fid_mri"].items():
        plot += k3d.points(
            np.array([coord], dtype=np.float32),
            color=fid_colors[name],
            point_size=FIDUCIAL_POINT_SIZE,
            name=name,
        )

    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    axis_defs = [
        (np.array([[AXIS_LENGTH, 0.0, 0.0]]), 0xFF0000, "X (R→L)"),
        (np.array([[0.0, AXIS_LENGTH, 0.0]]), 0x00FF00, "Y (P→A)"),
        (np.array([[0.0, 0.0, AXIS_LENGTH]]), 0x0000FF, "Z (I→S)"),
    ]
    for end_point, color, label in axis_defs:
        plot += k3d.line(np.vstack([origin, end_point]), color=color, width=AXIS_LINE_WIDTH, name=label)

    plot.camera = [0.0, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return plot


def _save_html(plot: k3d.Plot, save_path: Path, as_standalone: bool) -> None:
    """Save plot to HTML file, either standalone or snapshot-only."""
    if as_standalone:
        plot_json = json.dumps(plot.get_snapshot())
        html_content = STANDALONE_HTML_TEMPLATE.format(
            title=plot.name,
            k3d_version=k3d.__version__,
            height=DEFAULT_PLOT_HEIGHT,
            plot_json=plot_json,
        )
    else:
        html_content = plot.get_snapshot()

    with open(save_path, "w") as f:
        f.write(html_content)
    logger.info("Saved coregistration HTML to %s", save_path)


def plot_coregistration(
    fif_path: str,
    trans_file: str,
    head_surf_file: str,
    pipeline_metadata_file: str,
    add_brain: bool = False,
    lh_pial_file: Optional[str] = None,
    rh_pial_file: Optional[str] = None,
    save_to: Optional[str] = None,
    save_as_standalone_html: bool = False,
) -> k3d.Plot:
    """Interactive 3D coregistration quality check with optional HTML export.

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
    save_to : str or Path, optional
        If provided, save an HTML file of the visualization.
    save_as_standalone_html : bool
        If True and save_to is set, embed k3d JS library for offline viewing.
        If False, save only the plot JSON snapshot.

    Returns
    -------
    k3d.Plot
    """
    fif_path = Path(fif_path)
    trans_file = Path(trans_file)
    head_surf_file = Path(head_surf_file)
    pipeline_metadata_file = Path(pipeline_metadata_file)

    data = _load_coreg_data(fif_path, trans_file, head_surf_file, pipeline_metadata_file)

    lh_path = Path(lh_pial_file) if lh_pial_file else None
    rh_path = Path(rh_pial_file) if rh_pial_file else None
    plot = _build_plot(data, add_brain, lh_path, rh_path)

    if save_to is not None:
        _save_html(plot, Path(save_to), save_as_standalone_html)

    return plot
