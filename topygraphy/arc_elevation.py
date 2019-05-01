# arc_elevation.py
"""
Python program converting ARC ASCII data to x,y grid data saved as a Numpy .npz file.

Zachary Meves
6 April 2018

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
    """Class defining x, y, elevation data from an ARC ASCII conversion.

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
        Elevation data, indexed with (row, col) indices
    meshes : dict
        If generated, triangulations of the terrain, indexed by method name
    """

    def __init__(self, filename=None, resolution=30):
        """Constructor.

        Inputs
        ------
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

        Inputs
        ------
        filename : str
            Name of file to read"""

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

    def write(self, filename):
        """Write data to a Numpy .npz file"""

        print('Saving ElevData object to {} . . . '.filename, end='', flush=True)
        np.savez(filename, elev_data=self)
        print('Finished', flush=True)

    def _make_node_ids(self):
        """For internal use only. Make 2D array parallel to self._x_mesh and self._y_mesh storing IDs of nodes"""

        indices = np.zeros(self._x_mesh.shape)
        j = 0
        for i in np.ndindex(self._x_mesh.shape):
            indices[i] = j
            j += 1
        return indices

    def _make_coordinate_arrays(self):
        """For internal use. Generates x and y coordinate arrays from contained data.

        Outputs
        -------
        x : array of x coordinates
        y : array of y coordinates
        """

        num_x, num_y = self._data_xy.shape
        x = self._ll_coords[0] + np.arange(0, num_x, 1) * self._dx
        y = self._ll_coords[1] + np.arange(0, num_y, 1) * self._dx
        return x, np.flip(y, 0)

    @property
    def filename(self):
        return self._filename

    @property
    def x(self):
        return self._x_coords
    
    @property
    def y(self):
        return self._y_coords

    @property
    def elev_ij(self):
        return self._data_ij

    @property
    def elev_xy(self):
        return self._data_xy

    @property
    def meshes(self):
        return self._meshes

    def mesh(self, method):
        """Return mesh computed with method.

        Input
        -----
        mesh : str
            Method used to compute triangulation. One of:
                'delaunay'
                'split-cells'
            Append '_X' for meshes computed with downsample factor X
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
        """Compute elevation gradients"""
        # TODO: compute elevation gradients
        pass

    def _id_constant_gradient_regions(self):
        """Identify continuous regions of constant gradient"""
        # TODO: ID constant gradient regions
        pass

    ########################################################
    # Triangulation functions
    ########################################################

    def delaunay(self):
        """Creates Delaunay triangulation of elevation data."""

        x_array, y_array = np.ravel(self._x_mesh), np.ravel(self._y_mesh)  # Unroll grids

        points = np.zeros((len(x_array), 2))  # Generate n_points x 2 array
        points[:, 0] = x_array[:]
        points[:, 1] = y_array[:]
        
        print('Generating Delaunay triangulation for {} points . . . '.format(len(x_array)), flush=True, end='')
        tri = Delaunay(points)  # Do Delaunay triangulation
        print('Finished')

        self._meshes['delaunay'] = tri

    def split_cell(self):
        """Create split-cell triangulation of elevation data.

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
                elems[i, :]     = elem_ll
                elems[i + 1, :] = elem_ur
                i += 2

        # Create list of nodes
        points = np.empty((self._node_ids.size, 2))
        j = 0
        for i in np.ndindex(self._node_ids.shape):
            points[j, :] = [self._x_mesh[i], self._y_mesh[i]]
            j += 1
        
        print('Finished')

        tri = {'points' : points, 'simplices' : elems}
        self._meshes['split_cell'] = tri

    ###########################################################################
    # Visualization
    ###########################################################################

    def _set_labels(self, ax):
        """Sets plot labels appropriately"""

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def contour(self, *args, **kwargs):
        """Plots contours of elevation data (topographic map)"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111)
        p = ax.contour(self._x_coords, self._y_coords, self._data_ij, *args, **kwargs)

        self._set_labels(ax)
        c = fig.colorbar(p)
        c.set_label('Elevation')

        plt.show()

    def contourf(self, *args, **kwargs):
        """Plots filled contours of elevation data"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111)
        p = ax.contourf(self._x_coords, self._y_coords, self._data_ij, *args, **kwargs)
        self._set_labels(ax)

        c = fig.colorbar(p)
        c.set_label('Elevation')

        plt.show()

    def scatter(self, *args, **kwargs):
        """Plots 3D scatterplot of elevation data
        
        NOTE: This is likely to be extremely slow for large maps"""

        fig = plt.figure()  # Create figure
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(np.ravel(self._x_mesh), np.ravel(self._y_mesh), np.ravel(self._data_ij), *args, **kwargs)
        self._set_labels(ax)

        plt.show()

    def plot_surface(self, *args, **kwargs):
        """Plots elevation data as surface"""

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
        
        Input
        -----
        method : str
            Name of triangulation method"""

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

# if __name__ == '__main__':
#     """Run main program at command line.
#
#     Optional arguments are names of ARC ASCII files (.asc) to read and convert, and
#     optionally a .npz file to output results to"""
#
#     args = sys.argv[1:] # Get command line arguments
#     print('args = {}'.format(args))
#     converter = ARCConverter() # Create ARCConverter instance
#
#     # If no arguments given or help requested, print help string and exit
#     if len(args) == 0 or 'help' in [x.lower() for x in args]:
#         converter.print_help()
#     else:
#         input_filenames, output_filename = [], None
#         for arg in args:
#             if '.asc' in arg and arg[-4:] == '.asc':
#                 input_filenames.append(arg)
#             if '.npz' in arg and arg[-4:] == '.npz':
#                 output_filename = arg
#
#         if not output_filename:
#             output_filename = 'arc_ascii_converted.npz'
#
#         converter.output_file = output_filename
#         for fname in input_filenames:
#             converter.read(fname)
#
#         converter.write()
    
    

    

