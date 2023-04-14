#################################
# Test data processing pipeline #
#################################
import os

import pandas as pd
from tools.useful_functions import get_Steam_games, load_config

from scripts.data_processors.SteamDataProcessor import SteamProcessor


def format_steam_games(
    url: str, path_to_steam_games: str = None, recreate_test_cases: bool = True
) -> list:
    """Test the steam game processing pipeline

    Args:
        url (str): Scrape games from url
        path_to_steam_games (str, optional): Path where games are saved if they don't exist. Defaults to None.
        recreate_test_cases (bool, optional): Whether to export dataframe with processed data. Defaults to True.

    Returns:
        Newly created dataframe: (pd.DataFrame), old dataframe (pd.DataFrame): Two dataframes containing newly and priorly processed game names to be compared
    """
    steam_games = get_Steam_games(url=url, path_to_steam_games=path_to_steam_games)
    steam_processor = SteamProcessor(scraped_content=steam_games)
    steam_processor.write_scraped_data_into_dataframe()
    steam_processor.normalize_names("game", "game_normalized")
    steam_processor.dataframe = steam_processor.dataframe.iloc[:100, :]
    if recreate_test_cases:
        steam_processor.export_dataframe(
            steam_processor.dataframe,
            path=os.path.join(
                os.getcwd(),
                "tests/steam_games_test_data/steam_first_100_normalized.csv",
            ),
        )
    old_steam_df = steam_processor.load_dataframe(
        "tests/steam_games_test_data/steam_first_100_normalized.csv"
    )
    old_steam_df["app_id"] = pd.to_numeric(old_steam_df["app_id"])
    return steam_processor.dataframe, old_steam_df


def test_steam_processor():
    config = load_config("config.yaml")
    STEAM_URL = config["STEAM"]["URL"]
    STEAM_SAVED_DATA = config["STEAM"]["SAVED_DATA"]
    df_steam_new, df_steam_old = format_steam_games(
        url=STEAM_URL, path_to_steam_games=STEAM_SAVED_DATA, recreate_test_cases=True
    )
    assert all(
        df_steam_new == df_steam_old
    ), f"Something is wrong, dataframes differ in {df_steam_new[df_steam_new != df_steam_old]}."
