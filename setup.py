try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize



from polygonintegrals.cythonize_custom import mycythonize
# from Cython.Build import cythonize
import numpy as np

my_modules = mycythonize('polygonintegrals/clipSegmentOnGrid.py')
my_modules += mycythonize('polygonintegrals/integreWithinPolygon.py')
my_modules += mycythonize('polygonintegrals/xorshift.py')
my_modules += mycythonize('polygonintegrals/polygonClipping.py')
#my_modules = cythonize([r"*.pyx"],                        
                       #annotate=True,
                       #extra_compile_args=['/EHsc'])





import numpy as np

# Get the version number.
import runpy
__version_str__ = runpy.run_path("polygonintegrals/version.py")["__version_str__"]


import os
paths=['polygonintegrals/examples']
files_to_copy=[]
for path in paths:
	for (dir, _, files) in os.walk(path):
		for f in files:
			files_to_copy.append(os.path.join(dir[len('polygonintegrals')+1:], f))
print('found %d file to copy'%len(files_to_copy))


libname="polygonintegrals"
setup(
name = libname,
version= __version_str__,
packages=         ['polygonintegrals'],
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()],
package_data={'polygonintegrals':files_to_copy}
)

