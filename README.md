# uIPSF: Universal inverse modelling of point spread functions for SMLM localization and microscope characterization 
The point spread function (PSF) of a microscope describes the image of a point emitter. Knowing the accurate PSF model is essential for various imaging tasks, including single molecule localization, aberration correction or deconvolution. 

Here we present uiPSF (universal inverse modelling of Point Spread Functions), a toolbox to infer accurate PSF models either from image stacks of fluorescent beads or directly from images of single blinking fluorophores, the raw data in SMLM. It is a powerful tool to characterize and optimize a microscope as it reports the aberration modes, including field-dependent aberrations.  The resulting PSF model enables accurate 3D super-resolution imaging using single molecule localization microscopy.
Our modular framework is applicable to a variety of microscope geometries, and the forward model can include system specific characteristics such as the bead size, camera pixel size and transformations among channels. We demonstrate its application in single objective systems with single or multiple channels, 4Pi-SMLM, and lattice light-sheet microscopes.

## Overview
Follow the [installation](#installation) instructions closely to install uiPSF. 

To get started, open the notebooks in the [demo folder](demo), for instance the notebook for [single-channel PSFs](demo/demo_beadPSF_1ch.ipynb).

You can find data to use the demo notebooks [here](xxx).

## Systems tested
- Windows 11 with RTX 3080, RTX 2080
- Rocky Linux 8.7 with A6000

# Installation
## Windows
1. Install miniconda for windows, [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda Powershell Prompt, clone the PSFlearing package     
```
git clone https://github.com/ries-lab/PSFLearning.git
cd .\PSFLearning\
git checkout test_notebook
```
3. Create a new conda enviroment for PSFlearning package  
- for GPU: 
```
conda env create --name psfinv --file=environment.yml
```   
- for CPU: 
```
conda create --name psfinv python=3.7.10
```
4. Activate the installed enviroment and install the PSFlearning package
```
conda activate psfinv
pip install -e .
```

## Linux
1. Install miniconda for linux, [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal, clone the PSFlearing package     
```
git clone https://github.com/ries-lab/PSFLearning.git
cd .\PSFLearning\
git checkout test_notebook
```

3. Create a new conda enviroment for PSFlearning package  
- for GPU: 
```
conda env create --name psfinv --file=environment.yml
```   
- for CPU: 
```
conda create --name psfinv python=3.7.10
```
4. Activate the installed enviroment and install the PSFlearning package
```
conda activate psfinv
pip install -e .
```
5. If GPU version is installed, add cudnn path
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```



## Builds (for fitting)
Download (or upload) the builds for fitting here within the respective subfolder for your system here [OwnCloud builds](https://oc.embl.de/index.php/s/WrXnfC8Yj44iBq2).

