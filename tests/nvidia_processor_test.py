#################################
# Test data processing pipeline #
#################################
import os

from tools.useful_functions import get_Nvidia_games, load_config

from scripts.data_processors.NvidiaDataProcessor import NvidiaProcessor


def format_nvidia_games(
    url: str,
    classname: str,
    get_first_n_games: int = None,
    path_to_nvidia_games: str = None,
    recreate_test_cases: bool = True,
) -> list:
    """Test processing of scraped nvidia content

    Args:
        url (str): Link to nvidia games
        classname (str): Javascript token of game name
        get_first_n_games (int, optional): Limit scraping to first n games. Defaults to None.
        path_to_nvidia_games (str, optional): path to priorly scraped data. Defaults to None.
        recreate_test_cases (bool, optional): whether to scrape and save the scraped data. Defaults to True.

    Returns:
        Newly created dataframe: (pd.DataFrame), old dataframe (pd.DataFrame): Two dataframes containing newly and priorly processed game names to be compared
    """
    # Either load data or scrape again if file does not exists
    nvidia_games = get_Nvidia_games(
        url=url,
        classname=classname,
        path_to_nvidia_games=path_to_nvidia_games,
        get_first_n_games=get_first_n_games,
    )
    # Instantiate processor
    nvidia_processor = NvidiaProcessor(scraped_content=nvidia_games)
    # Separate game and platform names
    nvidia_processor.write_games_platforms_to_dataframe()
    # build normalized game name
    nvidia_processor.normalize_names("game", "game_normalized")
    # save scarped data
    if recreate_test_cases:
        nvidia_processor.export_dataframe(
            nvidia_processor.dataframe,
            path=os.path.join(
                os.getcwd(),
                "tests/nvidia_games_test_data/nvidia_first_100_normalized.csv",
            ),
        )
    # Return old and new data
    return nvidia_processor.dataframe, nvidia_processor.load_dataframe(
        "tests/nvidia_games_test_data/nvidia_first_100_normalized.csv"
    )


def test_nvidia_processor():
    # Load configurations
    config = load_config("config.yaml")
    NVIDIA_URL = config["NVIDIA"]["URL"]
    NVIDIA_CLASSNAME = config["NVIDIA"]["CLASSNAME"]
    NVIDIA_SAVED_DATA = os.path.join(
        os.getcwd(), "tests/nvidia_games_test_data/first_100_games.pkl"
    )
    GET_FIRST_N_GAMES = 100
    # Scarpe and load data
    df_nvidia_new, df_nvidia_old = format_nvidia_games(
        url=NVIDIA_URL,
        classname=NVIDIA_CLASSNAME,
        get_first_n_games=GET_FIRST_N_GAMES,
        path_to_nvidia_games=NVIDIA_SAVED_DATA,
        recreate_test_cases=True,
    )
    # test
    assert df_nvidia_new.equals(
        df_nvidia_old
    ), f"Something is wrong, dataframes differ."
