import os, sys, shutil

curdir = os.path.dirname(os.path.abspath(__file__))

shutil.rmtree(os.path.join(os.path.dirname(curdir), 'docs'))
os.system('sphinx-build -b html . ../docs')
