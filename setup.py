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
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Scientific/Engineering :: Physics"              
      ],

      scripts=['bin/learn_psf'],

      packages=['psflearning'], # TODO: put correct name here # what's actually imported by people later
      python_requires='>=3.9',
      install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "tensorflow",
            "tensorflow-probability",
            "scikit-image",
            "tqdm",
            "czifile",
            "hdfdict",
            "dotted_dict",
            "omegaconf",
            "pytest"
            #"NanoImagingPack"
            # TODO: right now it is not even possible to install NIP into an empty environment...
            # so either write Rainer and fixes this or get rid of these packages
            # TODO: what else?, add version numbers, test which versions work
      ],
      extras_require={  # can be used by pip install -e .[examples]
                        # then these additional packages are installed
            "examples": [
                  "ipykernel" # this seems not easy to be installed with pip
                  "napari",
                  # TODO: what else?, add version numbers, test which versions work
            ],
            "gpu": [
                  "tensorflow-gpu"
            ],
            "dev": [
                  "check-manifest",
                  "pytest"
                  # TODO: can be used for developers, only we needed if we are going to use any testing framework
            ]
      }
)