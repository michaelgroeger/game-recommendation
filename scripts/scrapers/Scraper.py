##########################
# Generic scraper class  #
##########################

import json
import pickle

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class Scraper:
    """Generic Scraper class that can be used to scrape different websites using a headless chrome browser."""

    def __init__(self, url: str = None) -> None:
        """Instantiates a Scraper class

        Args:
            url (str): url to scrape content from, Defaults to None
        """
        self.url = url

    def get_url_content(self):
        """Sends GET statement to webserver to retrieve the answer and convert it to JSON

        Returns:
            dict: Answer of GET request
        """
        return requests.get(self.url).json()

    def build_chrome_driver(self):
        """Builds a chrome browser with some options to be used in downstream tasks"""
        # Instantiate Options object to receive settings for browser
        options = Options()
        options.headless = True
        # Instantiate Serivce object that is responsible for starting and stopping chromedriver
        s = Service(ChromeDriverManager().install())
        # Instantiate the actual headless chrome
        driver = webdriver.Chrome(service=s, options=options)
        # Maximize window to local limits to minimize chances to miss some content.
        driver.maximize_window()
        self.driver = driver

    def pickle_data(self, path: str, object):
        """Saves object by pickeling

        Args:
            path (str): where to save object
            object: Object to be saved to path
        """
        with open(path, "wb") as f:
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def unpickle_data(self, path: str) -> object:
        """Load pickled object

        Args:
            path (str): path to object

        Returns:
            object: Anything that was pickled before
        """
        with open(path, "rb") as f:
            file = pickle.load(f)
        return file

    def save_data_as_json(self, path: str, dictionary: dict):
        """Saves object as json

        Args:
            path (str): where to save object
            dictionary (dict)
        """
        with open(path, "w") as f:
            json.dump(dictionary, f, indent=4)

    def load_json(self, path: str) -> object:
        """Load json object

        Args:
            path (str): path to object

        Returns:
            dict: Anything that was json before
        """
        with open(path, "r") as f:
            file = json.load(f)
        return file

    @staticmethod
    def export_dataframe(dataframe, path, save_parquet):
        if save_parquet == True:
            dataframe.to_parquet(path, index=False)
        else:
            dataframe.to_csv(path, index=False)

    @staticmethod
    def load_dataframe(path):
        return pd.read_csv(path)
