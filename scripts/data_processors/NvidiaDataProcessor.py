from re import search

import pandas as pd
from data_processors.DataProcessor import DataProcessor


class NvidiaProcessor(DataProcessor):
    """
    Processor for scraped content from Nvidia

    Args:
        DataProcessor (DataProcessor): Inherits from DataProcessor base class.
    """

    def __init__(self, dataframe: pd.DataFrame = None, scraped_content=None):
        # Call super function to have access to all attributes and methods from DataProcessor
        super().__init__(scraped_content)
        self.dataframe = dataframe

    def write_games_platforms_to_dataframe(self):
        # The scraped game name from Nvidia goes like that some_game (some_platform), we need to separate them
        platforms, games = self.separate_games_from_platform()
        self.dataframe = pd.DataFrame({"game": games, "platform": platforms}, dtype=str)

    def separate_games_from_platform(self):
        # Check if no platform is given
        platform = [
            NvidiaProcessor.no_platform_handler(game_string)
            for game_string in self.scraped_content
        ]
        game = [
            NvidiaProcessor.game_name_handler(game_string)
            for game_string in self.scraped_content
        ]
        return platform, game

    @staticmethod
    def no_platform_handler(game_string: str) -> str:
        """Extracts game name and handles exceptions

        Args:
            game_string (str): E.g. "some_game (some_platform)"

        Returns:
            str: Either None, when () not in string, or 'some_platform'
        """
        try:
            # Extracts platform if exists  "some_game (some_platform)" -> 'some_platform'
            return search("\(([^)]+)", game_string).group(1)
        # If fails check if there is no platform
        except AttributeError:
            print(
                f"No platform in {game_string}, checking if no () is given. Export None then"
            )
            try:
                if "(" and ")" not in game_string:
                    return None
            # Escalate game because usually there should be at least something
            except AttributeError:
                print(f"separation failed check game name {game_string}")

    @staticmethod
    def game_name_handler(game_string: str) -> str:
        """Extracts game name from "some_game (some_platform)"

        Args:
            game_string (str): "some_game (some_platform)"

        Returns:
            str: some_game
        """
        # Check if platform is given and extract game name
        if "(" and ")" in game_string:
            return game_string.split("(")[0][:-1]
        else:
            # Return raw string
            return game_string

    def export_nvidia_platforms(self, path: str):
        """Export unique platforms such that they can be inspected

        Args:
            path (str): Where to save dataframe
        """
        # Extract unique platforms
        input = [
            p for p in self.dataframe["platform"].unique().tolist() if type(p) != float
        ]
        # Save to dataframe and export
        pd.DataFrame({"platform": input}).to_csv(path, index=False)
