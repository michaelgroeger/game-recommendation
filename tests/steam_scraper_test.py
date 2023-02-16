######################
# Test steam scraper #
######################
import os

from scripts.scrapers.SteamScraper import SteamScraper
from scripts.tools.useful_functions import load_config


def get_steam_games(url: str) -> dict:
    """Returns output of Steam API

    Args:
        url (str): API endpoint

    Returns:
        dict, dict: app list of fresh answer, app list of prior runs
    """
    # Scrape first 100 steam games and load priorly saved ones
    steam_1 = SteamScraper(url=url)
    return steam_1.get_url_content()["applist"]["apps"][:100], steam_1.load_json(
        os.path.join(os.getcwd(), "tests/steam_games_test_data/first_100_games.json")
    )


def test_steam_scraper():
    config = load_config(os.path.join(os.getcwd(), "config.yaml"))
    STEAM_URL = config["STEAM"]["URL"]
    scraped_games, old_games = get_steam_games(url=STEAM_URL)
    # Process dictionaries so we can compare if they're equal, use zip to match entries in lists
    pairs = zip(scraped_games, old_games)
    assert any(x != y for x, y in pairs), f"Something is wrong, games differ"
