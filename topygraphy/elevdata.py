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

        indices = np.zeros(self._x_mesh.shape)
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

    def delaunay(self):
        """Creates Delaunay triangulation of elevation data. Computed mesh is stored under
        the name 'delaunay'."""

        x_array, y_array = np.ravel(self._x_mesh), np.ravel(self._y_mesh)  # Unroll grids

        points = np.zeros((len(x_array), 2))  # Generate n_points x 2 array
        points[:, 0] = x_array[:]
        points[:, 1] = y_array[:]
        
        print('Generating Delaunay triangulation for {} points . . . '.format(len(x_array)), flush=True, end='')
        tri = Delaunay(points)  # Do Delaunay triangulation
        print('Finished')

        self._meshes['delaunay'] = tri

    def split_cell(self):
        """Create split-cell triangulation of elevation data. Computed mesh is stored under
        the name 'split_cell'.

        Split-cell meshes split each square defined by the elevation grid into 
        two right triangles"""

        print('Generating split-cell triangulation for {} points . . . '.format(self._node_ids.size), 
              flush=True, end='')

        Nx = len(self._x_coords) - 1  # Number of x intervals
        Ny = len(self._y_coords) - 1  # Number of y intervals

        # Create element-to-node mapping
        num_elems = Nx * Ny * 2  # Number of elements total
        elems = np.empty((num_elems, 3))  # List of elements (arrays containing node IDs)

        # Define mapping from row, column of squares to corner nodes
        def _bl(r, c):
            return r * (Nx + 1) + c

        def _br(r, c):
            return _bl(r, c) + 1

        def _tr(r, c):
            return (r + 1) * (Nx + 1) + c + 1

        def _tl(r, c):
            return _tr(r, c) - 1

        # Iterate over elements, assigning indices of nodes
        i = 0
        for r in range(Nx):
            for c in range(Ny):
                tl, tr, br, bl = _tl(r, c), _tr(r, c), _br(r, c), _bl(r, c)
                elem_ll = [bl, br, tl]  # Lower-left element nodes
                elem_ur = [tr, tl, br]  # Upper-right element nodes
                
                # Assign elements to nodes
                elems[i, :] = elem_ll
                elems[i + 1, :] = elem_ur
                i += 2

        # Create list of nodes
        points = np.empty((self._node_ids.size, 2))
        j = 0
        for i in np.ndindex(self._node_ids.shape):
            points[j, :] = [self._x_mesh[i], self._y_mesh[i]]
            j += 1
        
        print('Finished')

        tri = {'points': points, 'simplices': elems}
        self._meshes['split_cell'] = tri

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
            try:
                p = ax.plot_trisurf(tri.points[:, 0], tri.points[:, 1], tri.simplices, np.ravel(self._data_ij), *args, **kwargs)
            except AttributeError:
                p = ax.plot_trisurf(tri['points'][:, 0], tri['points'][:, 1], tri['simplices'], np.ravel(self._data_ij), *args, **kwargs)

            self._set_labels(ax)

            if 'cmap' in kwargs:
                c = fig.colorbar(p)
                c.set_label('Elevation')

            plt.show()

        else:
            raise KeyError("No triangulation generated with method '{}'".format(method))

class ElevMesh:
    """Class defining an elevation data mesh."""

    # TODO: Implement

    pass

    

