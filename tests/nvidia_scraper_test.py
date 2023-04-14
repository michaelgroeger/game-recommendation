########################################
# Functions to test the nvidia scraped #
########################################
import os

from tools.useful_functions import load_config

from scripts.scrapers.NvidiaScraper import NvidiaScraper


def get_Nvidia_games_old_and_new(
    url: str,
    classname: str,
    get_first_n_games: int = None,
    recreate_test_cases: bool = True,
) -> list:
    """
    Scrapes games from NVIDIA_URL and compares them to priorly scraped ones

    Args:
        url (str): url where games are listed on NVIDIA webpage
        classname (str): javascript token which holds game information
        get_first_n_games (int, optional): Whether to limit the scraping to the first n games. Defaults to None.
        recreate_test_cases (bool, optional): Whether to recreate the test cases. Defaults to None. Useful when nvidia updated their website.
    Returns:
        scraped content: (pd.DataFrame), old content (pd.DataFrame): Two dataframes containing newly and priorly scraped games from Nvidia webpage
    """
    # instantiate scraper
    nvidia_1 = NvidiaScraper(
        url=url, classname=classname, get_first_n_games=get_first_n_games
    )
    nvidia_1.build_chrome_driver()
    # scrape games from website
    nvidia_1.get_games()
    # If recreate test case, save data
    if recreate_test_cases:
        nvidia_1.pickle_data(
            os.path.join(
                os.getcwd(), "tests/nvidia_games_test_data/first_100_games.pkl"
            ),
            nvidia_1.scraped_content,
        )
    # return new and old games
    return nvidia_1.scraped_content, nvidia_1.unpickle_data(
        os.path.join(os.getcwd(), "tests/nvidia_games_test_data/first_100_games.pkl")
    )


def test_nvidia_scraper():
    config = load_config("config.yaml")
    # URL to scrape from
    NVIDIA_URL = config["NVIDIA"]["URL"]
    # HTML Token to look out for
    NVIDIA_CLASSNAME = config["NVIDIA"]["CLASSNAME"]
    GET_FIRST_N_GAMES = 100
    scraped_games, old_games = get_Nvidia_games_old_and_new(
        NVIDIA_URL, NVIDIA_CLASSNAME, GET_FIRST_N_GAMES
    )
    # Compare only first 10 games because we just want to test that scraping is not broken
    assert scraped_games[:10] == list(
        old_games[:10]
    ), f"Something is wrong, games differ in {list(set(old_games) - set(scraped_games))}"
