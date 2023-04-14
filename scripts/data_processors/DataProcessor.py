import json

import pandas as pd


class DataProcessor:
    """
    Base class to handle and process scraped data
    """

    def __init__(self, scraped_content: list = []) -> None:
        super().__init__()
        self.scraped_content = scraped_content

    @staticmethod
    def export_data_to_csv(processed_data: list, colnames: list, path):
        """
        Takes in any kind of data, in our case processed game data,
        and saves them in a dictionary such as {'colname 1':processed_data[0], ...,'colname n':processed_data[n]}. Eventually data will be saved to csv.

        Args:
            processed_data (list): list of lists
            colnames (list): list of strings ['colname 1', 'colname 2', ...]
            path (string): where to export csv to
        """
        # build dictionary to insert data into
        insertion = {}
        try:
            if len(processed_data) == len(colnames):
                for i in range(len(processed_data)):
                    insertion[colnames[i]] = processed_data[i]
                df = pd.DataFrame(insertion)
                df.to_csv(path, index=False)
        except ValueError:
            print(f"both inputs need to be of same length")

    def save_data_as_json(self, path: str, dictionary: dict):
        """Saves object as json

        Args:
            path (str): where to save object
            dictionary (dict)
        """
        with open(path, "w") as f:
            json.dump(dictionary, f)

    def load_json(self, path: str) -> dict:
        """Load json object

        Args:
            path (str): path to object

        Returns:
            dict: Dictionary of json content
        """
        with open(path, "r") as f:
            file = json.load(f)
        return file

    @staticmethod
    def export_dataframe(
        dataframe: pd.DataFrame, path: str, save_parquet: bool = False
    ):
        """Saves dataframe either as csv or parquet

        Args:
            dataframe (pd.DataFrame, optional): Dataframe to be exported.
            path (str, optional): Path where dataframe should be exported to.
            save_parquet (bool, optional): Whether to save as parquet. Defaults to False, then save as csv.
        """
        if save_parquet == True:
            dataframe.to_parquet(path, index=False)
        else:
            dataframe.to_csv(path, index=False)

    @staticmethod
    def load_dataframe(path: str, parquet: bool = False) -> pd.DataFrame:
        """Loads dataframe from path

        Args:
            path (str): Path to dataframe
            parquet (bool, optional): Whether to load a parquet. Defaults to False, then load csv.

        Returns:
            pd.DataFrame: _description_
        """
        if parquet == True:
            pd.read_parquet(path)
        else:
            pd.read_csv(path)
        return pd.read_csv(path)

    @staticmethod
    def preprocess(input: str) -> str:
        """Normalizes string input. Concatenates, lowercases and drops everything that is not alphanumeric.

        Args:
            input (str): Some text

        Returns:
            str: Normalized text.
        """
        return "".join(word.lower() for word in str(input) if word.isalnum())

    def normalize_names(self, column_to_be_normalized: str, new_column_name: str):
        """Normalizes game names, but can be used on any string column.

        Args:
            column_to_be_normalized (str): Column in dataframe to operate on.
            new_column_name (str): Name of new column holding processed text.
        """
        self.dataframe[new_column_name] = self.dataframe[column_to_be_normalized].map(
            lambda s: self.preprocess(s)
        )
        self.dataframe.drop_duplicates(inplace=True)
