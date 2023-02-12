import pandas as pd
from data_processors.DataProcessor import DataProcessor
from tqdm import tqdm


class SteamProcessor(DataProcessor):
    """
    Processor for scraped content from Steam

    Args:
        DataProcessor (DataProcessor): Inherits from DataProcessor base class.
    """

    def __init__(self, dataframe: pd.DataFrame = None, scraped_content=None):
        # Call super function to have access to all attributes and methods from DataProcessor
        super().__init__(scraped_content)
        self.dataframe = dataframe

    def write_scraped_data_into_dataframe(self):
        list_of_games = self.scraped_content["applist"]["apps"]
        app_ids = []
        games = []
        for game in tqdm(list_of_games, desc="Writing Steam games to dataframe.."):
            app_ids.append(game["appid"])
            games.append(game["name"])
        self.dataframe = pd.DataFrame({"app_id": app_ids, "game": games}, dtype=str)
        self.dataframe["app_id"] = pd.to_numeric(self.dataframe["app_id"])
