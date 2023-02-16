# Game Recommendation Project

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
# installs git lfs
git lfs install
# pulls large files
git lfs pull
```

If you have any issues, please read the messages I provided in ```./install_environment.sh``` they provide pointers how the issue might be resolved.

For the following scripts please make sure the environment 'project' is activated in you terminal. If unsure please run:

```zsh
conda activate project
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

If you see a message like

```zsh
2023-02-16 11:48:06.772 `st.experimental_singleton` is deprecated...
```

Don't worry about that. Simply refresh the page or proceed to Recommendations and the message will be gone for this session.

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

They contain outputs from scraping the nvidia game website and the Steam API. Since these endpoints change on a regular basis or the IP Address got blocked the tests may fail. For this, I provided you data from prior runs to inspect.
If the test cases succeed and there is new data, the old files should have been overwritten. If you get an error such as: ```selenium.common.exceptions.WebDriverException: Message: unknown error: cannot find Chrome binary```, then check
out this stackoverflow post which might help: [Link](https://stackoverflow.com/questions/46026987/selenium-gives-selenium-common-exceptions-webdriverexception-message-unknown). Its possible linked to the fact that you don't have Chrome or
chromedriver installed.

## Closing Remarks

This repository was tested and worked in:

```zsh
MacOS: Version 12.5.1
Ubuntu: Version 22.04
```

During the tests it seemed that at least 8 GB of RAM should be available for everything to be running smoothly.
We hope you enjoy this work!
