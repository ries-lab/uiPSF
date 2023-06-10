from setuptools import setup

with open("README.md", "r") as f:
      long_descrition = f.read()

setup(
      name='psflearning', # TODO: put correct name here # name for the pip install command, not neccessarily the same as the name of the python code
      version='0.0.1',
      description='Learning sinlge- or multi-channel PSFs from data using Tensorflow.',
      long_descrition=long_descrition,
      long_descrition_content_type="text/markdown",

      url='https://git.embl.de/ries/psfmodelling',
      author='Jonas Hellgoth, Sheng Liu',
      author_email='jonas.hellgoth@embl.de, sheng.liu@embl.de',

      license='LICENSE.txt', # TODO: choose a license and put it in license.txt --> https://choosealicense.com/
      classifiers=[ # availabel on https://pypi.org/classifiers/
            "Development Status :: 2 - Pre-Alpha",
            "Environment :: GPU :: NVIDIA CUDA :: 11.2",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            # TODO: add license here
            "Natural Language :: English",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Scientific/Engineering :: Physics"              
      ],


      packages=['psflearning'], # TODO: put correct name here # what's actually imported by people later
      python_requires='>=3.7',
      install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "tensorflow==2.9.1",
            "tensorflow-probability==0.17",
            "scikit-image",
            "tqdm",
            "czifile",
            "hdfdict",
            "dotted_dict",
            "omegaconf",
            "ipykernel"
            # TODO: what else?, add version numbers, test which versions work
      ],
)