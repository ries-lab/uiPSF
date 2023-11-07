# uiPSF: Universal inverse modelling of point spread functions for SMLM localization and microscope characterization 
The point spread function (PSF) of a microscope describes the image of a point emitter. Knowing the accurate PSF model is essential for various imaging tasks, including single molecule localization, aberration correction or deconvolution. 

Here we present uiPSF (universal inverse modelling of Point Spread Functions), a toolbox to infer accurate PSF models either from image stacks of fluorescent beads or directly from images of single blinking fluorophores, the raw data in SMLM. It is a powerful tool to characterize and optimize a microscope as it reports the aberration modes, including field-dependent aberrations.  The resulting PSF model enables accurate 3D super-resolution imaging using single molecule localization microscopy.
Our modular framework is applicable to a variety of microscope geometries, and the forward model can include system specific characteristics such as the bead size, camera pixel size and transformations among channels. We demonstrate its application in single objective systems with single or multiple channels, 4Pi-SMLM, and lattice light-sheet microscopes.

# Systems tested
- Windows 11 with RTX 3080, RTX 2080
- Rocky Linux 8.7 with A6000

# Installation
## Windows
1. Install miniconda for windows, [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda Powershell Prompt, clone the uiPSF package     
```
git clone https://github.com/ries-lab/uiPSF.git
cd uiPSF
```
3. Create a new conda enviroment for the uiPSF package  
- for GPU: 
```
conda env create --name psfinv --file=environment.yml
```   
- for CPU: 
```
conda create --name psfinv python=3.7.10
```
4. Activate the installed enviroment and install the uiPSF package
```
conda activate psfinv
pip install -e .
```

## Mac
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Mac.
2. Open Terminal and follow the [installation for Windows](#Windows) to install the uiPSF package. Only the CPU version is supported. 

## Linux
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Linux.
2. Follow the [installation for Windows](#Windows) to install the uiPSF package.
3. If the GPU version is installed, add cudnn path
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

# Demo notebooks
- For bead data
  - [Single-channel PSF modelling](demo/demo_beadPSF_1ch.ipynb).
  - [Multi-channel PSF modelling](demo/demo_beadPSF_2ch.ipynb).
  - [4Pi PSF modelling](demo/demo_beadPSF_4pi.ipynb).
  - [PSF modelling from a lattice light-sheet microscope](demo/demo_beadPSF_1ch_LLS.ipynb)
  - [Field-dependent PSF modelling ](demo/demo_beadPSF_FD.ipynb).
- For SMLM data
  - [Single-channel PSF modelling](demo/demo_insituPSF_1ch.ipynb).
  - [Tetrapod PSF modelling](demo/demo_insituPSF_TP.ipynb).
  - [Multi-channel PSF modelling](demo/demo_insituPSF_2ch.ipynb).
  - [4Pi PSF modelling](demo/demo_insituPSF_4pi.ipynb).
  - [Field-dependent PSF modelling](demo/demo_insituPSF_FD.ipynb).
- Microscope characterization
  - [Evaluation of standard microscope systems](demo/demo_eval_system.ipynb).
  - [Evaluation of field-dependent aberration](demo/demo_eval_system_FD.ipynb).
- [Generate PSF model at a given imaging depth](demo/demo_genPSF.ipynb) 

# Example data 
- 40 nm bead data from single-channel, dual-color ratiometric and 4Pi systems.
- bead data from a single-channel system with a FOV of 177 um x 177 um.
- bead data from a lattice-light sheet microscope.
- SMLM data of Nup96-AF647 from a single-channel system with a FOV of 177 um x 177 um.
- SMLM data of tubulin-AF647 from a single-channel system with astigmatism aberration applied by a deformable mirror
- SMLM data of TOMM20-AF647 from a single-channel system with Tetrapod PSFs applied by a phase plate
- SMLM data of Nup96-AF647 and WGA-CF680 from a dual-color ratiometric system.
- SMLM data of Nup96-mMaple from a 4Pi-SMLM system

Download the [example data](https://zenodo.org/records/10027718)
# How to run demo notebook
1. Install uiPSF for your operating system.
2. Install [Visual Studio Code](https://code.visualstudio.com/Download).
3. Open Visual Studio Code (VScode), click *Extensions* from the sidebar menu and search for `Python` and install `Python extension for VScode`.
4. Go to File->Open Folder, select the uiPSF folder from git clone.
5. Open the file *demo/datapath.yaml*, change the `main_data_dir` to the path of the downloaded example data.
6. Navigate to a demo notebook, e.g. *demo/demo_beadPSF_1ch.ipynb*.
7. Click the run button of the first cell, if running for the first time, a window will popup asking to install the `ipykernel` package, click install. Then a drop down menu will show up asking to select the kernel, select the created conda enviroment `psfinv` during the installation.
8. Run subsequent cells sequentially.

For explanation of user defined parameters, please see list of [all user defined parameters](config/parameter%20description.md). 

## Tips
- If a GPU is not available, comment the last two lines in the first cell *Setup environment* of the demo notebook.
- Don't run two notebooks at the same time, click `Restart` at the top of the notebook to release the memory.

# Localization using SMAP and FD-DeepLoc
Tutorials for using the PSF model generated from uiPSF for localization analysis. 
- [Single channel SMLM imaging](tutorial/tutorial%20for%20fit_fastsimple.pdf).
- [Ratiometric dual-color SMLM imaging](tutorial/Tutorial%20for%20fit_global_dualchannel.pdf).
- [4Pi-SMLM imaging](tutorial/tutorial%20fit_4pi.pdf)
- [Single channel SMLM imaging with large FOV](tutorial/Tutorial%20for%20FD_aberrations.pdf)
# Need help?
Open an issue here on github, or contact Jonas Ries (jonas.ries@univie.ac.at)
