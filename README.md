# Setting up Conda Environment for Ray RLlib and PyTorch

This guide explains how to set up a Conda environment for working with Ray RLlib and PyTorch.

## Prerequisites

- Conda installed ([Download Conda](https://docs.conda.io/en/latest/miniconda.html))

## Steps

1. **Create a Conda Environment:**

   Open your terminal and run the following command to create a new Conda environment named `Whitening_Agent_Black_Box`:

   ```bash
   conda create --name Whitening_Agent_Black_Box python=3.9
   conda activate Whitening_Agent_Black_Box
   conda install nvidia::cuda
   conda install anaconda::cudatoolkit
   conda install -c conda-forge "ray-default"
   conda install pytorch::pytorch
   conda install conda-forge::opencv
   conda install anaconda::scikit-learn
   conda install conda-forge::matplotlib
   pip install gymnasium
   pip install gymnasium[all]
   pip install gymnasium[classic-control]
   pip install swig
   pip install gymnasium[box2d]
   pip install h5py
   conda install -c conda-forge "ray[rllib]"
   conda install bokeh
   conda install -c conda-forge altair-all
   ```
   
