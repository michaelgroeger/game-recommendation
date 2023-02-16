#!/bin/bash
echo "Creating a conda environment with Python 3.10.8 and IPython..."
conda create -n project python=3.10.8 ipython
if [ $? -eq 0 ]; then
  echo "Conda environment creation succeeded."
else
  echo "Conda environment creation failed. Do you have miniconda (https://docs.conda.io/en/latest/miniconda.html) or anaconda (https://docs.anaconda.com/anaconda/install/index.html) installed?"
  exit 1
fi
# Change to ~/Anaconda3/etc/profile.d/conda.sh in case you don't use miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project
if [ $? -eq 0 ]; then
  echo "Conda environment activated."
else
  echo "Conda environment activation failed. Please check if the environment project exists by running conda env list"
  exit 1

fi
echo "Installing conda-build..."
conda install conda-build
if [ $? -eq 0 ]; then
  echo "Conda-build installation succeeded."
else
  echo "Conda-build installation failed. Please refer to https://docs.conda.io/projects/conda-build/en/stable/install-conda-build.html"
  exit 1
fi

echo "Developing the scripts directory..."
conda develop ./scripts
if [ $? -eq 0 ]; then
  echo "Developing the scripts directory succeeded."
else
  echo "Developing the scripts directory failed. One alternative way to make the scripts available for import is to move the folders contained in scripts/ into the base directory of this repository."
  exit 1
fi

echo "Installing the requirements..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
  echo "Requirements installation succeeded."
else
  echo "Requirements installation failed. Please check requirements.txt and install the packages manually"
  exit 1
fi

echo "Installing git lfs to retrieve files"
git lfs install
if [ $? -eq 0 ]; then
  echo "Installation of git lfs suceeded"
  git lfs pull
else
  echo "Installation of git lfs failed. Please refer to https://git-lfs.com/ on how to install it. You need it to download large files such as models and data"
  exit 1
fi

echo "If you did not receive any error messages the repo should be ready."
