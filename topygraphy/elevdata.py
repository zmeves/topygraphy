"""
elevdata.py
Zachary Meves
6 April 2018

Data types to store elevation data obtained from ARC ASCII files.


See http://resources.esri.com/help/9.3/arcgisengine/com_cpp/GP_ToolRef/Spatial_Analyst_Tools/esri_ascii_raster_format.htm
for ARC ASCII format specification.
See https://en.wikipedia.org/wiki/Esri_grid for more information.
"""

import sys, os
import numpy as np
import numpy.ma as ma
from copy import deepcopy
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ElevData:
    """Class defining elevation vs. x, y data from an ARC ASCII conversion.

    Attributes
    ----------
    filename : str
        Name of source ARC ASCII file
    x : ndarray
        Array of x coordinates
    y : ndarray
        Array of y coordinates
    elev_xy : 2D array
        Elevation data, indexed with (x, y) indices
    elev_ij : 2D array
        Elevation data, indexed with (row, col) indices (transpose of elev_xy)
    meshes : dict
        If generated, triangulations of the terrain, indexed by triangulation method name
    meshgrids : tuple
        X and Y-coordinate meshgrids for x-y indexed elevation data
    node_ids : 2D array
        Array of node IDs for x-y indexed elevation data. Used in mesh definition.
    """

    def __init__(self, filename=None, resolution=30):
        """Constructor.

        Arguments
        ---------
        filename : str
            Optional, name of ARC ASCII file to read
        resolution : float
            Optional, resolution of ARC ASCII data in meters
        """

        # Initialize data
        self._filename = None
        self._data_ij = None
        self._data_xy = None
        self._ll_coords = None
        self._dx = None
        self._x_coords, self._y_coords = None, None
        self._x_mesh, self._y_mesh = None, None
        self._node_ids = None
        self._meshes = dict()  # Triangulations of elevation surface

        # Read file if supplied
        if filename:
            self.read_arc_ascii(filename, resolution)

    def read_arc_ascii(self, filename, resolution):
        """Reads ARC ASCII data stored in filename and populates internal data.

        Arguments
        ---------
        filename : str
            Name of file to read
        resolution : float
            Resolution of ARC ASCII data in meters"""

        with open(filename) as f:
            # Read first line (number of columns)
            l = f.readline().split()
            cols = int(l[1])
            # Read 2nd line (number of rows)
            l = f.readline().split()
            rows = int(l[1])
            # Read 3rd line (lower-left corner x value)
            l = f.readline().split()
            xll_corner = float(l[1])
            # Read 4th line (lower-left corner y value)
            l = f.readline().split()
            yll_corner = float(l[1])
            # Read 5th line (cell side length)
            l = f.readline().split()
            dx = float(l[1])
            # Read 6th line (NODATA value)
            l = f.readline().split()
            nodata_value = float(l[1])

            # Preliminary information has now been read.
            # Iterate over lines of file - each line defines one row

            data = np.zeros((rows, cols))  # row, col indexed elevation data
            i = 0
            for line in f:
                data[i, :] = [float(x) for x in line.split()]
                i += 1

            # Mask invalid data values
            data = ma.masked_equal(data, nodata_value)

            # Assign object data
            self._filename = filename
            self._data_ij = data
            self._data_xy = ma.transpose(data)
            self._ll_coords = (xll_corner, yll_corner)
            self._dx = resolution

            self._x_coords, self._y_coords = self._make_coordinate_arrays()
            self._x_mesh, self._y_mesh = np.meshgrid(self._x_coords, self._y_coords, indexing='ij')
            self._node_ids = self._make_node_ids()

    def write(self, filename, suppress=False):
        """Write data to a Numpy .npz file

        Arguments
        ---------
        filename : str
            Name of .npz file to write data to
        suppress : bool
            Optional, True to suppress print() output"""

        if not suppress:
            print('Saving ElevData object to {} . . . '.format(filename), end='', flush=True)
        np.savez(filename, elev_data=self)
        if not suppress:
            print('Finished', flush=True)

    def _make_node_ids(self):
        """For internal use only. Make 2D array parallel to self._x_mesh and self._y_mesh
        storing IDs of nodes for mesh generation.

        Returns
        -------
        indices : 2D array
            Node IDs, parallel to self._x_mesh and self._y_mesh"""

        indices = np.zeros(self._x_mesh.shape, dtype=int)
        j = 0
        for i in np.ndindex(self._x_mesh.shape):
            indices[i] = j
            j += 1
        return indices

    def _make_coordinate_arrays(self):
        """For internal use. Generates x and y coordinate arrays from contained data.

        Returns
        -------
        x : ndarray
            Array of x coordinates
        y : ndarray
            Array of y coordinates
        """

        num_x, num_y = self._data_xy.shape
        x = self._ll_coords[0] + np.arange(0, num_x, 1) * self._dx
        y = self._ll_coords[1] + np.arange(0, num_y, 1) * self._dx
        return x, np.flip(y, 0)

    @property
    def filename(self):
        """Return name of ARC ASCII source file"""
        return self._filename

    @property
    def x(self):
        """Return array of x coordinates"""
        return self._x_coords
    
    @property
    def y(self):
        """Return array of y coordinates"""
        return self._y_coords

    @property
    def elev_ij(self):
        """Return 2D array of elevation data in row, column order"""
        return self._data_ij

    @property
    def elev_xy(self):
        """Return 2D array of elevation data in x, y order"""
        return self._data_xy

    @property
    def meshgrids(self):
        """Return x-y indexed data's x and y-coordinate meshgrids"""
        return self._x_mesh, self._y_mesh

    @property
    def node_ids(self):
        """Return node IDs for x-y indexed data"""
        return self._node_ids

    @property
    def meshes(self):
        """Return dictionary of generated meshes, indexed by triangulation method"""
        return self._meshes

    def mesh(self, method):
        """Return mesh computed with triangulation method.

        Arguments
        ---------
        method : str
            Method used to compute triangulation. One of:
                'delaunay'
                'split-cells'
            Append '_X' for meshes computed with downsample factor X

        Returns
        -------
        mesh : ElevMesh
        """

        if method in self._meshes:
            return self._meshes[method]
        else:
            raise KeyError('No mesh computed with method "{}"'.format(method))

    def downsample(self):
        """Downsample elevation data, masking points that do not contribute to change in 
        curvature. 

        For example, for a collection of coplanar points (defining a plane), the interior points are masked 
        since they are not necessary to define the plane."""

        # TODO: implement downsampling
        pass

        # Compute elevation derivatives wrt x and y: gradient vectors for each point

        # Identify continuous regions with constant gradient

        # Mask interior points in each region (apply mask to elevation data and x, y meshgrids)

    def _compute_gradients(self):
        """For internal use only. Compute elevation gradients"""
        # TODO: compute elevation gradients
        pass

    def _id_constant_gradient_regions(self):
        """For internal use only. Identify continuous regions of constant gradient"""
        # TODO: ID constant gradient regions
        pass

    ########################################################
    # Triangulation functions
    ########################################################

    def split_cell(self):
        """Create split-cell triangulation of elevation data.

        Split-cell meshes split each square defined by the elevation grid into
        two right triangles. Computed mesh is stored under the name 'split-cell'.
        """

        self.compute_mesh('split-cell')

    def delaunay(self):
        """Creates Delaunay triangulation of elevation data. Computed mesh is stored under
        the name 'delaunay'."""

        self.compute_mesh('delaunay')

    def compute_mesh(self, method):
        """Compute triangulation of elevation data.

        Arguments
        ---------
        method : str
            Name of triangulation method. One of `ElevMesh.methods`"""

        if method not in ElevMesh.methods:
            raise NameError("{} is not a valid mesh computation method. Choose one of {}".format(
                method, ElevMesh.methods))
        if method not in self._meshes:
            self._meshes[method] = ElevMesh(method=method, elev_data=self)
        self._meshes[method].compute()

    ###########################################################################
    # Visualization
    ###########################################################################

    def _set_labels(self, ax):
        """For internal use only. Sets plot labels appropriately.

        Arguments
        ---------
        ax : matplotlib Axes
            Axes to set labels for"""

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def contour(self, *args, **kwargs):
        """Plots contours of elevation data (topographic map).

        Arguments
        ---------
        *args : positional arguments
            Positional arguments to pass to matplotlib.axes.contour(*args, **kwargs)
        **kwargs : keyword arguments
            Keyword arguments to pass to matplotlib.axes.contour(*args, **kwargs)"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111)
        p = ax.contour(self._x_coords, self._y_coords, self._data_ij, *args, **kwargs)

        self._set_labels(ax)
        c = fig.colorbar(p)
        c.set_label('Elevation')

        plt.show()

    def contourf(self, *args, **kwargs):
        """Plots filled contours of elevation data.

        Arguments
        ---------
        *args : positional arguments
            Positional arguments to pass to matplotlib.axes.contourf(*args, **kwargs)
        **kwargs : keyword arguments
            Keyword arguments to pass to matplotlib.axes.contourf(*args, **kwargs)"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111)
        p = ax.contourf(self._x_coords, self._y_coords, self._data_ij, *args, **kwargs)
        self._set_labels(ax)

        c = fig.colorbar(p)
        c.set_label('Elevation')

        plt.show()

    def scatter(self, *args, **kwargs):
        """Plots 3D scatterplot of elevation data
        
        NOTE: This is likely to be extremely slow for large maps.

        Arguments
        ---------
        *args : positional arguments
            Positional arguments to pass to matplotlib.axes.scatter(*args, **kwargs)
        **kwargs : keyword arguments
            Keyword arguments to pass to matplotlib.axes.scatter(*args, **kwargs)"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(np.ravel(self._x_mesh), np.ravel(self._y_mesh), np.ravel(self._data_ij), *args, **kwargs)
        self._set_labels(ax)

        plt.show()

    def _scale_z_axis(self, ax):
        """For internal use only. Scale z-axis in 3D plots to have uniform visual appearance."""
        # Cubic bounding box for equal aspect ratio fudging
        x_max, x_min = np.max(self._x_coords), np.min(self._x_coords)
        y_max, y_min = np.max(self._y_coords), np.min(self._y_coords)
        z_max, z_min = np.max(self._data_ij), np.min(self._data_ij)
        max_range = np.array([x_max - x_min,
                              y_max - y_min,
                              z_max - z_min]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_max + x_min)
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_max + y_min)
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_max + z_min)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    def plot_surface(self, *args, **kwargs):
        """Plots elevation data as surface.

        Arguments
        ---------
        *args : positional arguments
            Positional arguments to pass to matplotlib.axes.plot_surface(*args, **kwargs)
        **kwargs : keyword arguments
            Keyword arguments to pass to matplotlib.axes.plot_surface(*args, **kwargs)"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111, projection='3d')
        p = ax.plot_surface(self._x_mesh, self._y_mesh, self._data_xy, *args, **kwargs)
        ax.set_aspect('equal')

        self._scale_z_axis(ax)

        self._set_labels(ax)
        
        if 'cmap' in kwargs:
            c = fig.colorbar(p)
            c.set_label('Elevation')

        plt.show()

    def plot_trisurf(self, method, *args, **kwargs):
        """Plots elevation triangulation. Must have triangulation generated.
        
        Arguments
        ---------
        method : str
            Name of triangulation method to plot mesh for
        *args : positional arguments
            Positional arguments to pass to matplotlib.axes.plot_trisurf(*args, **kwargs)
        **kwargs : keyword arguments
            Keyword arguments to pass to matplotlib.axes.plot_trisurf(*args, **kwargs)
        """

        if method in self._meshes:
            tri = self._meshes[method]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # try:
            p = ax.plot_trisurf(tri.points[:, 0], tri.points[:, 1], tri.simplices, np.ravel(self._data_ij), *args, **kwargs)
            # except AttributeError:
            #     p = ax.plot_trisurf(tri['points'][:, 0], tri['points'][:, 1], tri['simplices'], np.ravel(self._data_ij), *args, **kwargs)

            self._scale_z_axis(ax)

            self._set_labels(ax)

            if 'cmap' in kwargs:
                c = fig.colorbar(p)
                c.set_label('Elevation')

            plt.show()

        else:
            raise KeyError("No triangulation generated with method '{}'".format(method))

class ElevMesh:
    """Class defining an elevation data mesh. Closely follows NumPy triangulation format"""

    methods = ['split-cell', 'delaunay']  # Method options

    def __init__(self, method=None, elev_data=None):
        """Constructor.

        Arguments
        ---------
        method : str
            Mesh computation method to use. One of:
                split-cell
                delaunay
        elev_data : ElevData
            ElevData object to create a mesh for"""

        self._method = method
        self._elev_data = elev_data
        self._tri = None
        self._nodes = None

        self._method_mapping = {'split_cell': self._split_cell,
                                'delaunay': self._delaunay}

    @property
    def method(self):
        """Return method name"""
        return self._method

    @method.setter
    def method(self, new_method):
        """Set the mesh computation method.

        Arguments
        ---------
        new_method : str
            New method to use. One of `methods`"""

        if new_method not in self.methods:
            print('{} is not a valid method. Choose one of {}'.format(new_method, self.methods))
        elif new_method != self._method:  # If setting a new method
            self._method = new_method
            # Wipe previously created data
            self._tri = None
            self._nodes = None

    @property
    def simplices(self):
        """Return triangles"""
        return self._tri

    @property
    def tri(self):
        """Return triangles"""
        return self._tri

    @property
    def points(self):
        """Return points"""
        return self._nodes

    @property
    def nodes(self):
        """Return points"""
        return self._nodes

    def compute(self):
        """Perform mesh computation with current method and ElevData object."""

        if self._elev_data is not None:  # Ensure an ElevData is associated
            try:
                self._method_mapping[self._method]()  # Do mesh computation
            except KeyError:
                raise KeyError("Unknown mesh computation type: {}".format(self._method))
        else:
            print("No ElevData is associated with this mesh object yet")

    def _split_cell(self):
        """Create split-cell triangulation of elevation data.

        Split-cell meshes split each square defined by the elevation grid into
        two right triangles"""

        print('Generating split-cell triangulation for {} points . . . '.format(self._elev_data.node_ids.size),
              flush=True, end='')

        node_ids = self._elev_data.node_ids

        Nx = len(self._elev_data.x) - 1  # Number of x intervals
        Ny = len(self._elev_data.y) - 1  # Number of y intervals

        # Create element-to-node mapping
        num_elems = Nx * Ny * 2  # Number of elements total
        elems = np.empty((num_elems, 3))  # List of elements (arrays containing node IDs)

        # Define mapping from row, column of squares to corner nodes
        def _bl(r, c):
            # return r * (Nx + 1) + c
            return r, c

        def _br(r, c):
            # return _bl(r, c) + 1
            return r + 1, c

        def _tr(r, c):
            # return (r + 1) * (Nx + 1) + c + 1
            return r + 1, c + 1

        def _tl(r, c):
            # return _tr(r, c) - 1
            return r, c + 1

        # Iterate over elements, assigning indices of nodes
        i = 0
        for r in range(Nx):
            for c in range(Ny):
                tl, tr, br, bl = _tl(r, c), _tr(r, c), _br(r, c), _bl(r, c)
                # elem_ll = [bl, br, tl]  # Lower-left element nodes
                # elem_ur = [tr, tl, br]  # Upper-right element nodes
                elem_ll = [node_ids[bl], node_ids[br], node_ids[tl]]
                elem_ur = [node_ids[tr], node_ids[tl], node_ids[br]]

                # Assign elements to nodes
                elems[i, :] = elem_ll
                elems[i + 1, :] = elem_ur
                i += 2

        # Create list of nodes
        points = np.empty((node_ids.size, 2))
        j = 0
        x_mesh, y_mesh = self._elev_data.meshgrids
        for i in np.ndindex(node_ids.shape):
            points[j, :] = [x_mesh[i], y_mesh[i]]
            j += 1

        print('Finished')

        self._nodes = points
        self._tri = elems

    def _delaunay(self):
        """Creates Delaunay triangulation of elevation data."""

        x_mesh, y_mesh = self._elev_data.meshgrids
        x_array, y_array = np.ravel(x_mesh), np.ravel(y_mesh)  # Unroll grids

        points = np.zeros((len(x_array), 2))  # Generate n_points x 2 array
        points[:, 0] = x_array[:]
        points[:, 1] = y_array[:]

        print('Generating Delaunay triangulation for {} points . . . '.format(len(x_array)), flush=True, end='')
        tri = Delaunay(points)  # Do Delaunay triangulation
        print('Finished')

        self._nodes = tri.points
        self._tri = tri.simplices
