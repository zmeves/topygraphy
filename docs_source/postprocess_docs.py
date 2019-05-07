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

print('Moving files from {} to {}'.format(sourcedir, targetdir))
try:
    files = os.listdir(sourcedir)  # All files in source directory
except FileNotFoundError:
    exit()

for f in files:
    shutil.move(os.path.join(sourcedir, f), targetdir)

# Remove source directory after all files have been moved
print('Removing directory {}'.format(sourcedir))
os.removedirs(sourcedir)

print('postprocess_docs.py finished')
