"""
Example #1
==========

Reading ARC ASCII file containing elevation data for Monch, Eiger, Jungfrau, and
Lauterbrunnen valley in the Bernese Alps, plot contour map, create and plot
split-cell.
"""

# Import and path updating
import sys, os
wd = os.path.abspath(__file__)  # Get absolute path of this file
# topygraphy is located at ../
sys.path.append(os.path.dirname(wd))

from topygraphy.elevdata import ElevData

###############################################################################
# Provide file name and read it
###############################################################################

input_filename = 'jungfrau_arc_ascii.asc'  # Name of input file (same folder)
jungfrau = ElevData(input_filename)  # Create ElevData object based on file

###############################################################################
# Create filled contour plot with 100 color levels
###############################################################################
jungfrau.contourf(100)
jungfrau.plot_surface()
###############################################################################
# Compute a split-cell mesh and plot the resulting mesh along with a surface
# of the elevatoin data
###############################################################################
# jungfrau.split_cell()
jungfrau.delaunay()

jungfrau.plot_trisurf('delaunay')
