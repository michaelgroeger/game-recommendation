# data-science-project

## Getting started

To have the easiest setup experience make sure the following bash scripts are executable:

```zsh
install_environment.sh
reproduce_benchmark_results.sh
start_app.sh
```

If unsure or if they are not executable yet, please run the following commands in your terminal:

```zsh
chmod 777 install_environment.sh
chmod 777 reproduce_benchmark_results.sh
chmod 777 start_app.sh
```

## Installing the environment

If you want the automated process then please run this command in your terminal:

```zsh
install_environment.sh
```

In the case you want to do it manually you can also execute the commands in this order:

```zsh
# Creates new conda environment
conda create -n project python=3.10.8 ipython
# Activates conda environment
conda activate project
# Makes script folder callable as module in python scripts
conda install conda-build
conda develop ./scripts
# Installs dependencies
pip install -r requirements.txt
```

## Reproduce results

To reproduce the benchmark results please run in your terminal

```zsh
reproduce_benchmark_results.sh
```

## Bring App online

To start the App locally, run

```zsh
start_app.sh
```

## Showcase Scraping pipeline

To showcase the scraping scripts, simply run

```zsh
start_app.sh
```
