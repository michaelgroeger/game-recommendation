##########################
# Nvidia scraper class   #
##########################
from scrapers.Scraper import Scraper
from selenium.webdriver.common.by import By
from tqdm import tqdm


class NvidiaScraper(Scraper):
    """Scrapes Nvidia game website to collect games

    Args:
        Scraper (class): Baseclass which is inherited
    """

    def __init__(self, url: str, classname: str, get_first_n_games: int = None):
        """Initialize Nvidia Scraping Object

        Args:
            url (str): url to get data from e.g. https://www.nvidia.com/de-de/geforce-now/games/
            classname (str): JavaScript token which holds game names e.g. game-name
            get_first_n_games (int, optional): Limit scraper request for testing. Defaults to None.
        """
        # Call to super function to have access to all attributes and methods of Scraper
        super().__init__(url)
        self.classname = classname
        self.get_first_n_games = get_first_n_games

    def get_games(self):
        # get javascript of website
        self.driver.get(self.url)
        # search javascript for content of interest which is marked by classname
        game_content = self.driver.find_elements(By.CLASS_NAME, self.classname)
        # extract all game content into list
        if self.get_first_n_games == None:
            games = [
                game.text
                for game in tqdm(game_content, desc="Getting games from NVIDIA")
            ]
        # Extract only first n games for testing purposes
        else:
            games = []
            for i, game in tqdm(
                enumerate(game_content),
                total=self.get_first_n_games,
                desc=f"Getting first {self.get_first_n_games} games from NVIDIA",
            ):
                games.append(game.text)
                if i == self.get_first_n_games:
                    break
        # close chrome session
        self.scraped_content = games
        self.driver.quit()
