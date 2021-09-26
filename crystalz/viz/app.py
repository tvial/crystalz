"""
Visualization application for the preprocessing methods

The app lets the user select a geometry file from a list, choose a preprocessing
method and tweak its parameters. The result displayed is a side-by-side view of:
- the component (atoms)
- the result of preprocessing

The app expects the data directory on the command line (`--xyz-dir` argument). All
`*.xyz` files directly under that directory are proposed for selection.

The application uses modules from parent packages, so it cannot be run simply with the
`streamlit run crystalz/viz/app.py` verb. The correct syntax is:

```
python -m streamlit.cli run crystalz/viz/app.py -- --xyz-dir /your/xyz/directory
```

(`--` tells Streamlit that the following arguments are handled by the app and not by the CLI)
"""

# pylint: disable=too-many-locals, too-many-arguments

import glob
import os
import argparse
from typing import cast, List, AnyStr, Callable, Tuple

import streamlit as st
import ipyvolume as ipv
import ipywidgets.embed as wembed
import pyvista as pv
import numpy as np

from crystalz.io.xyz import read_xyz
from crystalz import preprocessing


ATOM_COLORS = {
    'JMol': {
        'Al': (0.7490196078431373, 0.6509803921568628, 0.6509803921568628),
        'Ga': (0.7607843137254902, 0.5607843137254902, 0.5607843137254902),
        'In': (0.6509803921568628, 0.4588235294117647, 0.45098039215686275),
        'O': (1.0, 0.050980392156862744, 0.050980392156862744)
    },
    'Custom': {
        'Al': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        'Ga': (1.0, 0.4980392156862745, 0.054901960784313725),
        'In': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        'O': (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    }
}
"""
Atom colors for the structure plot.
- JMol: well-known color scheme, taken from http://jmol.sourceforge.net/jscolors/
- Custom: "Home-made" with Matplotlib colors, in which Al, Ga and In are easier to tell apart
"""


VOXEL_COLOR = [0.8, 0.7764705882352941, 0.7098039215686275]
"""
Color of the voxels for the Monochrome transfer function.
In this mode the values of the potential only change the opacity of the voxels
"""


@st.cache # type: ignore[misc]
def get_xyz_files(directory: str) -> List[str]:
    """
    Gets a sorted list of all `*.xyz` files under a directory (no recursion)

    Parameters
    ----------
    directory: str
        Path where to look for the files

    Returns
    -------
    List[str]
        Sorted list of file names (base names)
    """
    filenames = glob.glob(os.path.join(directory, '*.xyz'))
    basename = cast(Callable[[str], AnyStr], os.path.basename)
    return sorted(map(basename, filenames))


@st.cache # type: ignore[misc]
def get_monochrome_transfer_function() -> ipv.TransferFunction:
    """
    An alternative transfer function, suited for simpler compounds

    Returns
    -------
    ipv.TransferFunction
        A TF with a single color and a linear opacity ramp
    """
    n_levels = 256
    tf_data = np.zeros((n_levels, 4))
    tf_data[:, [0, 1, 2]] = VOXEL_COLOR
    tf_data[:, 3] = np.linspace(0, .05, n_levels)
    return ipv.TransferFunction(rgba=tf_data.astype(np.float32))


def run_app(directory: str) -> None:
    """
    Entry point of the application

    Parameters
    ----------
    directory: str
        Path where to look for XYZ files
    """
    filenames = get_xyz_files(directory)
    filename = st.sidebar.selectbox(
        'File', filenames,
        help='Name of the XYZ file to load and display'
    )

    st.sidebar.write('## Structure')
    color_schemes = sorted(ATOM_COLORS.keys())
    color_scheme = st.sidebar.selectbox(
        'Color scheme', color_schemes,
        help='How to color the kinds of atoms'
    )

    st.sidebar.write('## Voxels')
    method_names = preprocessing.METHODS.keys()
    method_name = st.sidebar.selectbox(
        'Method', method_names,
        help='Voxel calculation method'
    )

    xyz_max = st.sidebar.slider(
        'X/Y/Zmax', 1, 30, value=10,
        help='Maximum X, Y and Z coordinates of the cube'
    )

    resolution = st.sidebar.slider(
        'Voxel resolution', 16, 128, step=16,
        help='Number of voxels on each axis of the cube'
    )

    transfer_function = st.sidebar.selectbox(
        'Transfer function', ['Default', 'Monochrome'],
        help='The transfer function to render the voxels with'
    )

    xyz_path = os.path.abspath(os.path.join(directory, filename))
    with open(xyz_path, 'r', encoding='utf-8') as xyz_file:
        atoms, vectors = read_xyz(xyz_file)

    n_atoms = len(atoms[0])
    st.title(f'{filename} - {n_atoms} atoms')

    st.write('## Structure')

    render_structure(atoms, vectors, color_scheme)

    st.write('## Voxels')

    render_voxels(
        method_name,
        atoms, vectors,
        resolution,
        transfer_function,
        xyz_max, xyz_max, xyz_max
    )


def render_structure(
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray],
    vectors: np.ndarray,
    color_scheme: str
) -> None:
    """
    Renders the compound structure (atoms) in the browser

    Parameters
    ----------
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Kinds + atom centers + their radii
    vectors: np.ndarray
        Matrix formed by the vectors of the unit cell
    color_scheme: str
        Name of the color scheme (see `ATOM_COLORS`)
    """
    plotter = pv.Plotter(notebook=True)
    plotter.set_background('white')

    axes_properties = {
        'scale': 'auto',
        'tip_length': .03,
        'tip_radius': .01,
        'shaft_radius': .005
    }
    for vector in vectors:
        plotter.add_mesh(pv.Arrow(direction=vector, **axes_properties), color='red')

    for kind, center, radius in zip(*atoms):
        plotter.add_mesh(
            pv.Sphere(radius, center),
            color=ATOM_COLORS[color_scheme][kind],
        )

    plotter.reset_camera_clipping_range()
    scene = plotter.show(jupyter_backend='pythreejs', return_viewer=True)

    html = wembed.embed_snippet(scene)
    st.components.v1.html(f'<html><body>{html}</body></html>', width=1000, height=600)


def render_voxels(
    method_name: str,
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray],
    vectors: np.ndarray,
    resolution: int,
    tf_name: str,
    x_max: float,
    y_max: float,
    z_max: float
) -> None:
    """
    Calls the preprocesing method and renders the voxels in the browser. Here we allow
    x_max != y_max != z_max, although the app will only let the user tune one value.

    Parameters
    ----------
    method_name: str
        Preprocessing method to use
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Kinds + atom centers + their radii
    vectors: np.ndarray
        Matrix formed by the vectors of the unit cell
    resolution: int
        Number of voxels in each direction
    tf_name: str
        Name of the transfer function to use ('Default', or 'Monochrome')
    x_max: float
        Maximum X coordinate of the cube
    y_max: float
        Maximum Y coordinate of the cube
    z_max: float
        Maximum Z coordinate of the cube
    """
    voxels = preprocessing.METHODS[method_name].get_voxels( # type: ignore[attr-defined]
        atoms,
        vectors,
        resolution,
        x_max, y_max, z_max
    )

    ipv.figure()

    if tf_name == 'Monochrome':
        transfer_function = get_monochrome_transfer_function()
    else:
        # None is ipyvolume's default
        transfer_function = None

    ipv.volshow(
        voxels,
        tf=transfer_function,
        extent=[[0, x_max], [0, y_max], [0, z_max]]
    )

    html = wembed.embed_snippet(ipv.gcc())
    st.components.v1.html(f'<html><body>{html}</body></html>', width=1000, height=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--xyz-dir', type=str, required=True,
        help='Directory where to find the XYZ files'
    )
    args = parser.parse_args()

    run_app(args.xyz_dir)
