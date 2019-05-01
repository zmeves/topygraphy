"""
ARC to STL runner
"""
from arc_elevation import ElevData

input_filename, name = 'Jungfrau/output_srtm.asc', 'Jungfrau'
jungfrau = ElevData(input_filename)

# ac.grid(name).contour(100)
jungfrau.contourf(100)
# ac.grid(name).scatter()

# ac.grid(name).plot(100, mode='contour')
# ac.grid(name).delaunay()
jungfrau.split_cell()

jungfrau.plot_surface()
jungfrau.plot_trisurf('split_cell')
