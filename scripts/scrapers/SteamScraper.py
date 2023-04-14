##########################
#  Steam scraper class   #
##########################
import json
import os
import pickle
import re
import warnings
from time import sleep
from xml.etree import ElementTree as etree

import pandas as pd
import xmltodict
from lxml import etree
from requests import get
from scrapers.Scraper import Scraper
from tqdm import tqdm
from unidecode import unidecode

warnings.simplefilter(action="ignore", category=FutureWarning)
CLEANR = re.compile("<.*?>")


class SteamScraper(Scraper):
    """Will scrape all game data from Steam

    Args:
        Scraper (class): Baseclass which is inherited
    """

    def __init__(self, url: str = None) -> None:
        super().__init__(url)

    def get_game_information(
        self,
        app_ids,
        continue_from=None,
        step_save: int = None,
        save_parquet: bool = True,
        proxies=None,
        buggy_apps: list = [],
        buggy_apps_path: str = None,
    ):
        df = pd.DataFrame(
            columns=[
                "appid",
                "name",
                "type",
                "num_recommendations",
                "website",
                "header_image",
                "developers",
                "publishers",
                "required_age",
                "controller_support",
                "detailed_description",
                "short_description",
                "about_the_game",
                "supported_languages",
                "genres",
                "categories",
                "price",
                "currency",
                "metacritic_score",
                "release_date" "coming_soon",
            ]
        )
        if continue_from != None:
            app_ids = app_ids[continue_from:]
        for idx, id in tqdm(
            enumerate(app_ids),
            total=len(app_ids),
            desc="Collecting game information",
        ):
            id = str(id)
            if id in buggy_apps:
                continue
            if idx % 200 == 0 and idx != 0:
                print("Wrap up session to not get blocked, switch to next ip")
                self.game_information = df
                # read big one
                all_metadata = pd.read_parquet(
                    os.path.join(os.getcwd(), "data/steam_games_metadata.parq")
                )
                all_metadata = pd.concat(
                    [all_metadata, self.game_information], axis=0, ignore_index=True
                )
                self.export_dataframe(
                    all_metadata,
                    os.path.join(os.getcwd(), "data/steam_games_metadata.parq"),
                    save_parquet=save_parquet,
                )
                with open(buggy_apps_path, "wb") as fp:
                    pickle.dump(buggy_apps, fp)
                return True
            if id == "":
                continue
            params = {
                "appids": id,
            }
            if proxies == None:
                res = get(
                    "https://store.steampowered.com/api/appdetails", params=params
                ).json()
            else:
                res = get(
                    "https://store.steampowered.com/api/appdetails",
                    params=params,
                    proxies=proxies,
                ).json()
            # try:
            if res == None:
                print(f"Can't curl {id}")
                print(f"Got res = {res}")
                buggy_apps.append(id)
                continue
            else:
                try:
                    content = res[id]["data"]
                except:
                    print(f"Problem with app id {id}, can't get data, will skip")
                    buggy_apps.append(id)
                    continue
                row = {
                    "appid": id,
                    "name": content["name"],
                    "type": content["type"],
                    "short_description": content["short_description"],
                    "website": content["website"],
                    "header_image": content["header_image"],
                    "publishers": content["publishers"],
                    "detailed_description": str(content["detailed_description"]),
                    "short_description": str(content["short_description"]),
                    "about_the_game": str(content["about_the_game"]),
                    "release_date": content["release_date"]["date"],
                    "coming_soon": content["release_date"]["coming_soon"],
                }
                if "required_age" in content:
                    if content["required_age"] == "17+":
                        row["required_age"]: 18
                    else:
                        row["required_age"] = int(content["required_age"])
                if "developers" in content:
                    row["developers"] = content["developers"]
                else:
                    row["developers"] = None
                if "genres" in content:
                    row["genres"] = content["genres"]
                else:
                    row["genres"] = None
                if "categories" in content:
                    row["categories"] = content["categories"]
                else:
                    row["categories"] = None
                if "recommendations" in content:
                    row["num_recommendations"] = content["recommendations"]["total"]
                else:
                    row["num_recommendations"] = None
                if "metacritic" in content:
                    row["metacritic_score"] = content["metacritic"]["score"]
                else:
                    row["metacritic_score"] = None
                if "controller_support" in content:
                    row["controller_support"] = content["controller_support"]
                else:
                    row["controller_support"] = None
                if "supported_languages" in content:
                    row["supported_languages"] = content["supported_languages"]
                else:
                    row["supported_languages"] = None
                if "price_overview" in content:
                    row["price"] = content["price_overview"]["final"]
                    row["currency"] = content["price_overview"]["currency"]
                else:
                    row["price"] = None
                if "genres" in content:
                    row["genre"] = [genre["description"] for genre in content["genres"]]
                else:
                    row["genre"] = None
                new_df = pd.DataFrame([row])
                df = pd.concat([df, new_df], axis=0, ignore_index=True)
                if step_save != None and idx % step_save == 0:
                    self.game_information = df
                    self.export_dataframe(
                        self.game_information,
                        os.path.join(
                            os.getcwd(), "data/steam_games_metadata_temp.parq"
                        ),
                        save_parquet=save_parquet,
                    )

    @staticmethod
    def process_review(text: str) -> str:
        """Input text and return cleaned version. Used on reviews.

        Args:
            text (str): Review text of game from user

        Returns:
            str: cleaned text
        """
        cleantext = (
            text.replace('"', "")
            .replace("\n", " ")
            .replace("\r", "")
            .replace("..", "")
            .replace("*", "")
        )
        cleantext = cleantext.strip('"')
        # remove html tokens
        cleantext = re.sub(CLEANR, " ", cleantext)
        # remove links
        cleantext = re.sub(r"http\S+", "", cleantext)
        # remove special characters
        cleantext = re.sub(r"[^\x00-\x7F]+", "", cleantext)
        # remove whitespace if present more than once
        cleantext = re.sub(" +", " ", cleantext)
        # Decode unicode
        cleantext = unidecode(cleantext)
        # remove parantheses
        cleantext = re.sub(r"[()]", "", cleantext)
        return cleantext

    @staticmethod
    def check_if_query_was_successfull(query):
        if query["success"] == 1:
            return True
        else:
            False

    def process_reviews(self, query_result: dict, df: pd.DataFrame, app_id: str):
        """Process answer of GET request of the reviews

        Args:
            query_result (dict): Dictonary of answer for GET request
            df (pd.DataFrame): Dataframe collecting the query results
            app_id (str): current app id

        Returns:
            df(pd.DataFrame): input DataFrame with appended results
            cursor(str): Cursor to paginate to next page
        """
        try:
            for i in query_result["reviews"]:
                row = {
                    "app_id": app_id,
                    "recommendationid": i["recommendationid"],
                    "author_id": i["author"]["steamid"],
                    "num_games_owned": i["author"]["num_games_owned"],
                    "num_reviews": i["author"]["num_reviews"],
                    "playtime_forever": i["author"]["playtime_forever"],
                    "playtime_last_two_weeks": i["author"]["playtime_last_two_weeks"],
                    "playtime_at_review": i["author"]["playtime_at_review"],
                    "language": i["language"],
                    "review": str(self.process_review(str(i["review"]))),
                    "timestamp_created": i["timestamp_created"],
                    "voted_up": i["voted_up"],
                    "votes_up": i["votes_up"],
                    "votes_funny": i["votes_funny"],
                    "weighted_vote_score": i["weighted_vote_score"],
                    "comment_count": i["comment_count"],
                    "steam_purchase": i["steam_purchase"],
                    "received_for_free": i["received_for_free"],
                }
                new_df = pd.DataFrame([row])
                df = pd.concat([df, new_df], axis=0, ignore_index=True)
        except:
            f"got bad query result for {query_result}"
            return df, None
        cursor = query_result["cursor"]
        return df, cursor

    def get_game_reviews(
        self, app_ids: list, continue_from: int = None, step_save: int = 10
    ):
        """Collects reviews of games. Important to get more users and their playtimes since there is no direct way to get
        users of steam.

        Args:
            app_ids (list): list of appids to query reviews from
            continue_from (int, optional): Continue from idx in appids. Defaults to None.
            step_save (int, optional): Save the data every n steps. Defaults to 10.
        """
        # Where reviews will be exported to
        review_path = os.path.join(os.getcwd(), "data/reviews.parq")
        # Build dataframe to collect information of reviewers
        df = pd.DataFrame(
            columns=[
                "app_id",
                "recommendationid",
                "author_id",
                "num_games_owned",
                "num_reviews",
                "playtime_forever",
                "playtime_last_two_weeks",
                "playtime_at_review",
                "language",
                "review",
                "timestamp_created",
                "voted_up",
                "votes_up",
                "votes_funny",
                "weighted_vote_score",
                "comment_count",
                "steam_purchase",
                "received_for_free",
            ]
        )
        if continue_from != None:
            app_ids = app_ids[continue_from:]
        # get first page of reviews for app
        for idx, id in tqdm(
            enumerate(app_ids),
            total=len(app_ids),
            desc="Collecting game information. Pausing every 15 calls for 10 seconds and every 200 for 60 seconds..",
        ):
            if idx % 15 == 0 and idx != 0:
                sleep(10)
            elif idx % 200 == 0 and idx != 0:
                sleep(60)
                print("Pausing 60 seconds")
            # Sleep this for every request
            sleep(0.25)
            # Params for scraper
            params = {"json": "1", "cursor": "*", "num_per_page": 100}
            # Send get request
            res = get(
                f"https://store.steampowered.com/appreviews/{id}", params=params
            ).json()
            # Process answer
            df, cursor = self.process_reviews(query_result=res, df=df, app_id=id)
            # No reviews
            if cursor == None:
                print(f"Will skip game {id}")
                continue
            # Paginating through reviews of current game
            runs = 0
            pbar = tqdm(
                total=runs + 1, desc=f"Working on game {id} with cursor {cursor}"
            )
            # Collect cursors
            cursors = []
            # Whild there was an answer and the cursor is not yet known
            while (
                self.check_if_query_was_successfull(res) == True
                and cursor not in cursors
            ):
                cursors.append(cursor)
                # Sleep to be nice to the endpoint
                sleep(0.25)
                # Request next 100 reviews and page
                params = {"json": "1", "cursor": f"{cursor}", "num_per_page": 100}
                res = get(
                    f"https://store.steampowered.com/appreviews/{id}", params=params
                ).json()
                df, cursor = self.process_reviews(res, df, app_id=id)
                pbar.update(1)
            pbar.close()
            # If step_save is activated then export dataframe every n steps
            if step_save != None and idx % 10:
                df.drop_duplicates(inplace=True)
                self.app_reviews = df
                self.app_reviews.to_parquet(review_path)
        # End of scraping
        df.drop_duplicates(inplace=True)
        self.app_reviews = df
        self.app_reviews.to_parquet(review_path)

    def query_player_and_games(self, user_id: str, proxies=None):
        """Queries a player and their games

        Args:
            user_id (str): Some players steam id
            proxies (dict, optional): Dict of proxies to use. Defaults to None.

        Returns:
            user_game_info (dict): Dictionary of games of player
        """
        # Send request
        params = {
            "tab": "all",
            "xml": "1",
        }
        if proxies == None:
            res = get(
                f"https://steamcommunity.com/profiles/{user_id}/games", params=params
            )
        else:
            res = get(
                f"https://steamcommunity.com/profiles/{user_id}/games",
                params=params,
                proxies=proxies,
            )
        # Process answer which comes in xml
        string_xml = res.content
        try:
            tree = etree.fromstring(string_xml)
            user_game_info = xmltodict.parse(
                etree.tostring(tree, pretty_print=True).decode()
            )
            return user_game_info
        except:
            print(f"Got error for id {user_id}, will skip")
            return None

    def wrap_up_session(
        self,
        user_games,
        no_games_or_private_user,
        buggy_users,
        path_user_games_previous_runs: str = "data/user_games.json",
        path_user_games_last_run: str = "data/user_games.json",
        buggy_users_path: str = "data/user_games_no_game_or_private.txt",
    ):
        """Export data that got collected that far

        Args:
            user_games_path (dict): Current query results
            user_games: User and game information so far collected
            no_games_or_private_user: User which are either having no games or which are set to private
            buggy_users: Users which are already know to be private or for which scraping fails in general
            path_user_games_previous_runs (str): Path where the data of all previous merged runs is saved
            path_user_games_last_run (str): Path where the data of the very last run is saved
            buggy_users_path (str): Path to text file with user ids of users for which API call fails
        """
        # Get path to data
        user_games_merged_path = os.path.join(
            os.getcwd(), path_user_games_previous_runs
        )
        user_games_path = os.path.join(os.getcwd(), path_user_games_last_run)
        buggy_users_path = os.path.join(os.getcwd(), buggy_users_path)
        # Export user game information to json
        self.save_data_as_json(user_games_path, user_games)
        # Add newly arrived faulty users to buggy users
        new_buggy_users = [
            user for user in no_games_or_private_user if user not in buggy_users
        ]
        # Load data from previous runs
        with open(
            user_games_merged_path,
            "r",
        ) as f:
            previously_scraped_data = json.load(f)
        print(
            f"There are {len(list(previously_scraped_data.keys()))} users in previously_scraped_data"
        )
        # Load data from last run
        with open(
            user_games_path,
            "r",
        ) as f:
            data_from_last_scraping_run = json.load(f)
        # Join previously_scraped_data with newly scraped data
        all = {}
        for key, value in previously_scraped_data.items():
            if key not in all.keys():
                all[key] = value
        for key, value in data_from_last_scraping_run.items():
            if key not in all.keys():
                all[key] = value
        print(f"There are {len(list(all.keys()))} users in previously_scraped_data")
        # Export new complete data and buggy users to be picked up by future runs
        with open(
            user_games_merged_path,
            "w",
        ) as f:
            json.dump(all, f, indent=4)
        with open(
            buggy_users_path,
            "a",
        ) as f:
            for line in new_buggy_users:
                f.write(f"{line}\n")

    def get_player_and_games(
        self,
        user_ids: list,
        continue_from: int = None,
        step_save: int = 25,
        path: str = "data/user_games.json",
        buggy_users: list = [],
        proxies: dict = None,
    ):
        """Will send get request to get players game list. Will save to json because we don't know yet how many games to be expected
        so we can't write it into a dataframe

        Args:
            user_ids (list): list of user ids to parser
            continue_from (int, optional): User to continue querying from. Defaults to None.
            step_save (int, optional): Whether to save the user info every x users to a file.
            path (str, optional): Where to save user games
            buggy_users (list, optional): Whether to save the user info every x users to a file.
            proxies (dict, optional): Whether to save the user info every x users to a file.
        """
        print(f"Step save is {step_save}")
        if continue_from != None:
            user_ids = user_ids[continue_from:]
        user_games = {}
        user_games_path = os.path.join(os.getcwd(), path)
        # get first page of reviews
        no_games_or_private_user = []
        for idx, id in tqdm(
            enumerate(user_ids),
            total=len(user_ids),
            desc=f"Collecting games from users",
        ):
            # Save current results
            if idx % step_save == 0:
                self.save_data_as_json(user_games_path, user_games)
            if idx == 300:
                print("Wrap up session to prevent blocking, switch to next ip")
                self.wrap_up_session(
                    user_games_path=user_games_path,
                    user_games=user_games,
                    no_games_or_private_user=no_games_or_private_user,
                    buggy_users=buggy_users,
                )
                return True
            # Query games of user
            query_result = self.query_player_and_games(user_id=id, proxies=proxies)
            if query_result == None:
                continue
            # Check if gameslist is present in result
            if "gamesList" in query_result.keys():
                # check if got an error and if yes which one
                if "error" in query_result["gamesList"].keys():
                    # Skip profile if set to private
                    if query_result["gamesList"]["error"] == "This profile is private.":
                        no_games_or_private_user.append(id)
                        continue
                    # Skip profile if no games present
                    elif (
                        query_result["gamesList"]["error"]
                        == "You've made too many requests recently. Please wait and try your request again later."
                    ):
                        print("Got blocked")
                        self.wrap_up_session(
                            user_games_path=user_games_path,
                            user_games=user_games,
                            no_games_or_private_user=no_games_or_private_user,
                            buggy_users=buggy_users,
                        )
                        return False
                elif query_result["gamesList"]["games"] == None:
                    no_games_or_private_user.append(id)
                    continue
            user_games[id] = query_result
