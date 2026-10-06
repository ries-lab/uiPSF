from setuptools import setup, find_packages

with open("README.md", "r") as f:
      long_descrition = f.read()

setup(
      name='psflearning', 
      version='0.0.1',
      description='A versatile and modular toolbox that uses inverse modelling to extract accurate PSF models for most SMLM imaging modalities from bead and single-molecule data.',
      long_descrition=long_descrition,
      long_descrition_content_type="text/markdown",

      url='https://github.com/ries-lab/uiPSF.git',
      author='Sheng Liu,Jonas Hellgoth, Jianwei Chen',
      author_email='shengliu@unm.edu, jonas.hellgoth@embl.de, 12149038@mail.sustech.edu.cn',

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


      packages=find_packages(include=['psflearning', 'psflearning.*']), 
      python_requires='>=3.7',
      install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "tensorflow>=2.9",
            # [tf] brings tf-keras, which tensorflow-probability needs with
            # TensorFlow >= 2.16 (Keras 3)
            "tensorflow-probability[tf]>=0.17",
            "h5py",
            "pillow",
            "scikit-image",
            "tqdm",
            "czifile",
            "dotted_dict",
            "omegaconf",
            "ipykernel"
            
      ]
)