# ERPO: Expert-Regularized Policy Optimization for Fine-Tuning SE(3)-Equivariant Diffusion Models in Drug Design


## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda and Pip
```bash
conda create -n erpo python=3.8
conda activate erpo
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
The code should work with PyTorch >= 1.9.0 and PyG >= 2.0. You can change the package version according to your need.

## Training
### Training 
```bash
python source/scripts/train_EGPO.py 
```
### Get model checkpoint
```bash
python ERPO/source/scripts/get_diffsmol_from_egpo.py --egpo_ckpt {trained_ERPO_model} --save_path {diffsmol_save_path}
```

## Sampling
### Sampling for pockets in the testset
```bash
python ERPO/source/scripts/sample_crossdocked_test_set.py
```
