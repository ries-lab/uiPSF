# uiPSF: Universal inverse modelling of point spread functions for SMLM localization and microscope characterization 
The point spread function (PSF) of a microscope describes the image of a point emitter. Knowing the accurate PSF model is essential for various imaging tasks, including single molecule localization, aberration correction or deconvolution. 

Here we present uiPSF (universal inverse modelling of Point Spread Functions), a toolbox to infer accurate PSF models either from image stacks of fluorescent beads or directly from images of single blinking fluorophores, the raw data in SMLM. It is a powerful tool to characterize and optimize a microscope as it reports the aberration modes, including field-dependent aberrations.  The resulting PSF model enables accurate 3D super-resolution imaging using single molecule localization microscopy.
Our modular framework is applicable to a variety of microscope geometries, and the forward model can include system specific characteristics such as the bead size, camera pixel size and transformations among channels. We demonstrate its application in single objective systems with single or multiple channels, 4Pi-SMLM, and lattice light-sheet microscopes.

**Reference**: [Liu S, Chen J, Hellgoth J, et al. Universal inverse modelling of point spread functions for SMLM localization and microscope characterization. Preprint. bioRxiv. 2023](https://doi.org/10.1101/2023.10.26.564064)

# System requirements
## Hardware
uiPSF can run on both CPU and GPU, however, we recommend installing the GPU version for fast processing speed. To install the GPU version, a GPU card that supports CUDA 11.2 is required. Reference to [Systems tested](#Systems-tested) for selecting your GPU card.
## Software
### OS supported
uiPSF is supported for Windows, Linux and MacOS. Only CPU version is supported for MacOS.
### Package dependencies
```base
cudatoolkit (GPU version only)
cudnn (GPU version only)
pip
python
numpy
scipy
matplotlib
tensorflow
tensorflow-probability
scikit-image
tqdm
czifile
hdfdict
dotted_dict
omegaconf
ipykernel
```
## Systems tested
- Windows 11 with RTX 4090, RTX 3080,RTX 3090, RTX 2080
- Windows 10 with RTX 4000, RTX 3090, RTX 1070
- Rocky Linux 8.7 with RTX A6000
- Ubuntu 20.04 with RTX 1070

# Installation
## Windows
Installation time for the GPU version is around 10 minutes.
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
2. Install uiPSF package.
- For TensorFlow 2.9
      
  Follow the [installation for Windows](#Windows) to install the uiPSF package.
  
- For lastest TensorFlow (Note that TensorFlow later than 2.10 is no longer supported on Window)
   
   a. Modify the version numbers in the *environment.yml* file as follows:
   ```
   - cudatoolkit=11.8
   - cudnn=8.4
   - python=3.9
   ```
   b. Remove the version numbers in `install_requires` in the *setup.py* file as follows:
   ```
   "tensorflow"
   "tensorflow-probability"
   ```
   c. Follow the [installation for Windows](#Windows) to install the uiPSF package.
     
   d. If the GPU version is intalled, run the following command
   ```
   pip install tensorflow[and-cuda]
   ```
   We used above procedure to intall uiPSF on a Linux computer with RTX A6000 to fully utilize the computability from the GPU.    
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
## Run time of learning the PSF models listed in the demos
The following run times are obtained from a desktop PC with Windows 11, RTX 3080.
|**PSF type**| run time (min)| # of parameters|
|:------------------|:----------------|:------------|
|**1ch LLS voxel**|1.9 | 31,144|
|**1ch zernike_vector**|0.5 | 992|
|**2ch zernike_vector**|5.1 | 3,827|
|**4pi zernike**|2.8 | 775|
|**FD zernike_vector**|16.1 | 98,680|
|**1ch *in situ***|4.7 | 10,433|
|**2ch *in situ***|13.1 | 22,404|
|**4pi *in situ***|35 | 35,189|
|**FD *in situ***|49.7 | 143,023|
# Example data 
- 40 nm bead data from single-channel, dual-color ratiometric and 4Pi systems.
- bead data from a single-channel system with a FOV of 177 um x 177 um.
- bead data from a lattice-light sheet microscope.
- SMLM data of Nup96-AF647 from a single-channel system with a FOV of 177 um x 177 um.
- SMLM data of tubulin-AF647 from a single-channel system with astigmatism aberration applied by a deformable mirror
- SMLM data of TOMM20-AF647 from a single-channel system with Tetrapod PSFs applied by a phase plate
- SMLM data of Nup96-AF647 and WGA-CF680 from a dual-color ratiometric system.
- SMLM data of Nup96-mMaple from a 4Pi-SMLM system

Download the [example data](https://zenodo.org/doi/10.5281/zenodo.8267520)
# How to run demo notebook
1. Install uiPSF for your operating system.
2. Install [Visual Studio Code](https://code.visualstudio.com/Download).
3. Open Visual Studio Code (VScode), click *Extensions* from the sidebar menu and search for `Python` and install `Python extension for VScode`.
4. Go to File->Open Folder, select the uiPSF folder from git clone.
5. Open the file *demo/datapath.yaml*, change the `main_data_dir` to the path of the downloaded example data.
6. Navigate to a demo notebook, e.g. *demo/demo_beadPSF_1ch.ipynb*.
7. Click the run button of the first cell, if running for the first time, a window will popup asking to install the `ipykernel` package, click install. Then a drop down menu will show up asking to select the kernel, select the created conda enviroment `psfinv` during the installation.
    - In case there is no window popup, an alternative method is: install `Jupyter` from *Extensions*, then click *Select Kernel* at the upper right corner of the demo notebook and select the `psfinv` from the dropdown menu.
9. Run subsequent cells sequentially.

- For explanation of user defined parameters and details of creating config files, please refer to [user defined parameters](config/parameter%20description.md).     
- For explanation of the output parameters from PSF learning, please refer to [output parameters](demo/Description%20of%20output%20parameters.md).
## Tips

- Please ensure that the computer's current graphics card driver supports CUDA 11.2.
- Don't run two notebooks at the same time, click `Restart` at the top of the notebook to release the memory.

# Localization using SMAP and FD-DeepLoc
Tutorials for using the PSF model generated from uiPSF for localization analysis. Use one of the [demo notebooks](#Demo-notebooks) to generate the corresponding PSF model (.h5 file) before using the following tutorials. 
- [Single channel SMLM imaging](tutorial/tutorial%20for%20fit_fastsimple.pdf).
- [Ratiometric dual-color SMLM imaging](tutorial/Tutorial%20for%20fit_global_dualchannel.pdf).
- [4Pi-SMLM imaging](tutorial/tutorial%20fit_4pi.pdf)
- [Single channel SMLM imaging with large FOV](tutorial/Tutorial%20for%20FD_aberrations.pdf)
# Need help?
Open an issue here on github, or contact Jonas Ries (jonas.ries@univie.ac.at)
