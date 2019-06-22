"""postprocess_docs.py

After running Sphinx, output files will be in ../docs/html
For GitHub pages, they should be in ../docs. This script just moves
all files from ../docs/html to ../docs and deletes the ../docs/html
directory."""

import os
import sys
import shutil

curdir = os.path.dirname(os.path.abspath(__file__))
sourcedir = os.path.join(os.path.dirname(curdir), 'docs', 'html')
targetdir = os.path.dirname(sourcedir)
tempdir = os.path.join(os.path.dirname(targetdir), 'temp_docs')
codedir = os.path.join(os.path.dirname(targetdir), 'topygraphy')

print('Moving files from {} to {}'.format(sourcedir, tempdir))
shutil.move(sourcedir, tempdir)

print('Deleting files from {}'.format(targetdir))
shutil.rmtree(targetdir)

print('Moving files from {} to {}'.format(tempdir, targetdir))
shutil.move(tempdir, targetdir)

# try:
#     files = os.listdir(sourcedir)  # All files in source directory
# except FileNotFoundError:
#     exit()

# for f in files:
#     shutil.move(os.path.join(sourcedir, f), targetdir)

# # Remove source directory after all files have been moved
# print('Removing directory {}'.format(sourcedir))
# os.removedirs(sourcedir)

print('Running sphinx-apidoc')
os.system('sphinx-apidoc -o {} {}'.format(os.path.join(targetdir, 'source'), codedir))

print('postprocess_docs.py finished')
