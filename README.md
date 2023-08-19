# PSFLearning
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

