# data-science-project

## Getting started

To have the easiest setup experience make sure the following bash scripts are executable:

```zsh
./install_environment.sh
./reproduce_benchmark_results.sh
./start_app.sh
./run_scraping_tests.sh
```

If unsure or if they are not executable yet, please run the following commands in your terminal:

```zsh
chmod 700 ./install_environment.sh
chmod 700 ./reproduce_benchmark_results.sh
chmod 700 ./start_app.sh
chmod 700 ./run_scraping_tests.sh
chmod 700 ./train_recommender.sh
```

## Installing the environment

If you want the automated process then please run this command in your terminal:

```zsh
./install_environment.sh
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
./reproduce_benchmark_results.sh
```

or run directly in your terminal:

```zsh
./python evaluate_recommender.py
```

## Train a recommender

If you want to train a recommender you have a rich set of options to do so. Please check the flags in ```train_recommender.py``` or run a sample training by simply executing

```zsh
./train_recommender.sh
```

## Bring App online

To start the App locally, run

```zsh
./start_app.sh
```

## Test Scraping pipeline

To test the scraping scripts, simply run

```zsh
./run_scraping_tests.sh
```

You can then inspect the files in:

```zsh
./tests/nvidia_games_test_data
./tests/steam_games_test_data
```

They contain some outputs from the scraping of the nvidia game website and the Steam API. Since these endpoints change on a regular basis the test may fail. For this I provided you data from prior runs to inspect.
If the test cases succeed and there is new data the old files should have been overwritten.
