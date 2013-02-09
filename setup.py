# See http://packages.python.org/distribute/setuptools.html
from setuptools import setup, find_packages

setupargs = {
    'name'          :       'cartesiancoords',
    'version'       :      	"0.0.0",  # will be replaced below
    'description'   :       'Cartesian coordinate systems and transformations',
    'fullname'      :       'cartesiancoords',
    'author'        :     	"Marshall Perrin",
    'author_email'  :      	"mperrin@stsci.edu",
    'url'           :  		"http://www.stsci.edu/~mperrin/software/cartesiancoords",
    'download_url'           :  		"http://www.stsci.edu/~mperrin/software/cartesiancoords-0.0.0.tar.gz",  # will be replaced below
    'platforms'     :      	["Linux","Mac OS X", "Win"],
    'requires'      :       ['numpy', 'matplotlib'],
    'packages'      :       ['cartesiancoords'],
    'test_suite'    :       'cartesiancoords.tests.run_tests',
    'classifiers'   :   [
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Development Status :: 4 - Beta"
        ],
    'long_description': """
Cartesian coordinate systems and transformations. 

This package allows users to define any number of distinct coordinate systems,
which may be translated or rotated arbitrarily with respect to one another,
and allows conversion of points, vectors, and rotations between those systems. 

It also provides some simple visualization capabilities for the above, using
matplotlib.draw3d.

"""
    }

# read in the version number from the code itself, following 
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

# don't actually import the _version.py, for the reasons described on that web page. 
import re
VERSIONFILE="cartesiancoords/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    setupargs['version'] = mo.group(1)
    setupargs['download_url'] = "http://www.stsci.edu/~mperrin/software/cartesiancoords/cartesiancoords-"+mo.group(1)+".tar.gz"
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))







# Now actually call setup

setup( **setupargs)
