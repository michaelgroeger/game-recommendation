#!/bin/bash
echo "Creating a conda environment with Python 3.10.8 and IPython..."
conda create -n project python=3.10.8 ipython
if [ $? -eq 0 ]; then
  echo "Conda environment creation succeeded."
else
  echo "Conda environment creation failed."
  exit 1
fi
# Change to ~/Anaconda3/etc/profile.d/conda.sh in case you don't use miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project
if [ $? -eq 0 ]; then
  echo "Conda environment activated."
else
  echo "Conda environment activation failed."
  exit 1

fi
echo "Installing conda-build..."
conda install conda-build
if [ $? -eq 0 ]; then
  echo "Conda-build installation succeeded."
else
  echo "Conda-build installation failed."
  exit 1
fi

echo "Developing the scripts directory..."
conda develop ./scripts
if [ $? -eq 0 ]; then
  echo "Developing the scripts directory succeeded."
else
  echo "Developing the scripts directory failed."
  exit 1
fi

echo "Installing the requirements..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
  echo "Requirements installation succeeded."
else
  echo "Requirements installation failed."
  exit 1
fi