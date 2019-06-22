import os, sys, shutil

curdir = os.path.dirname(os.path.abspath(__file__))

try:
    shutil.rmtree(os.path.join(os.path.dirname(curdir), 'docs'))
except FileNotFoundError:
    pass
os.system('sphinx-build -b html . ../docs')
