#####################################################################
# Processes games from Nvidia & Steam to extract the relevant data  #
#####################################################################
import os
import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from data_processors.DataProcessor import DataProcessor
from tqdm import tqdm
from unidecode import unidecode

# From: https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
# as per recommendation from @freylis, compile once only
CLEANR = re.compile("<.*?>")


class AllDataProcessor(DataProcessor):
    """Class that takes in both Nvidia and Steam data and processes it for downstream tasks.

    Args:
        DataProcessor (DataProcessor): Inherits from base class
    """

    def __init__(
        self,
        scraped_content: list = [],
        steam_dataframe: pd.DataFrame = None,
        nvidia_dataframe: pd.DataFrame = None,
        all_dataframe: pd.DataFrame = None,
    ) -> None:
        """Initializes Class

        Args:
            scraped_content (list, optional): Attribute inherited from based class but not needed here. Defaults to []
            steam_dataframe (pd.DataFrame, optional): Dataframe containing steam games. Defaults to None.
            nvidia_dataframe (pd.DataFrame, optional): Dataframe containing Nvidia Games. Defaults to None.
            all_dataframe (pd.DataFrame, optional): Dataframe containing data from Steam and/or Nvidida to be further processed once the data is already joined. Defaults to None.
        """
        super().__init__(scraped_content)
        self.steam_dataframe = steam_dataframe
        self.nvidia_dataframe = nvidia_dataframe
        self.all_dataframe = all_dataframe
        self.relevant_platforms = [
            "Steam",
            "Epic Games Store",
            "Origin",
            "Ubisoft Connect",
        ]

    def join_nvidia_steam_dataframe(self):
        """
        Join nvidia and steam dataframe creating all data Frame
        """
        # Merge on normalized game name
        self.joined_data = self.nvidia_dataframe.merge(
            self.steam_dataframe, on="game_normalized", how="left"
        )
        # Fill NaN in appid with zero which is the default appid for faulty apps now
        self.joined_data["app_id"].fillna(0, inplace=True)
        # Set appropiate types for joined data
        self.joined_data.astype(
            {
                "game_x": str,
                "platform": str,
                "game_normalized": str,
                "app_id": int,
                "game_y": str,
            }
        )
        # Enforce once more on appid because of a bug it didn't set the type before
        self.joined_data["app_id"] = self.joined_data["app_id"].astype(int)
        # Fill NaN with Unknown in Platform
        self.joined_data["platform"].fillna("Unknown", inplace=True)
        # Add more data to joined table by calling the corresponding methods
        # Extract all platforms from table
        self.get_platforms()
        self.encode_platforms()
        self.get_steam_only_games()
        self.get_mismatches()
        self.steam_only_games = pd.concat([self.steam_only_games, self.mismatches])
        self.drop_rows_being_resolved_but_still_having_zero_app_id()

    def get_platforms(self):
        """
        Extracts the platforms from the joined table once Nvidia and Steam data got joined
        """
        # Get unique platforms
        platforms = self.joined_data["platform"].unique()
        # seperate platforms if there are more than one in one field
        self.game_platforms = [i.strip().split(",") for i in platforms]
        self.game_platforms = list(
            set(
                [
                    item.lstrip()
                    for sublist in self.game_platforms
                    for item in sublist
                    if item in self.relevant_platforms
                ]
            )
        )

    def encode_platforms(self):
        """
        Will create a new column per platform in the joined table, then adds a one in the column of those platforms in which the game is available. In essence
        produces a one-hot-encoded like structure for the platforms.
        """
        # add new columns
        for i in self.game_platforms:
            self.joined_data[i] = 0
        df = self.joined_data.copy()
        # check if a platform is in list, then add 1 to relevant column
        for platform in self.game_platforms:
            mask = df["platform"].str.contains(platform)
            df.loc[mask, platform] = 1
        self.joined_data = df

    def get_steam_only_games(self):
        """
        Will return those games which are available on Steam.
        """
        self.steam_only_games = self.joined_data[self.joined_data["Steam"] == 1].copy()

    def get_mismatches(self):
        """
        Triggers resolving stage between mismatched Nvidia and Steam games due to small name mismatches e.g. some
        unicode character in the such as ® (Registered)
        """
        self.mismatches = self.steam_only_games[self.steam_only_games["app_id"] == 0]
        self.resolve_mismatches()

    def resolve_mismatches(self):
        """
        Resolves game name mismtached between steam games and Nvidia games
        """
        # get all titles from mismatch
        games = self.mismatches["game_normalized"].tolist()
        # Check if close matches file already exists if not genereate it
        close_matches_json = Path(os.path.join(os.getcwd(), "data/close_matches.json"))
        data_folder = Path(os.path.join(os.getcwd(), "data"))
        # If data folder doesn't exist then create it
        if not data_folder.is_dir():
            os.mkdir(data_folder)
        if not close_matches_json.is_file():
            # Noise words are words that may be added to the Steam API game name which causes a mismatch with the raw Nvidia Game Name
            noise_words = [
                "demo",
                "directors",
                "director",
                "cut",
                "edition",
                "deluxe",
                "gameoftheyear",
                "definitive",
                "premium",
                "digital",
                "reloaded",
                "steam",
                "digital",
            ]
            # Drop noise words
            for word in noise_words:
                games = [game.replace(word, "") for game in games]
            # convert steam games to string
            steam_games = [
                str(i) for i in self.steam_dataframe["game_normalized"].tolist()
            ]

            results = {}
            # Ignore those matches in the raw Steam data that contain those names because they are not relevant for the app such as sountracks
            ignore_strings = [
                "trailer",
                "justcause2preorder",
                "tsereloaded",
                "demo",
                "soundtrack",
            ]
            # Get closes matches in pool of Steam games and save to file
            for game in tqdm(games, desc="Resolving mismatches"):
                # Get n close matches which have a higher similarity score than 0.7
                matches = get_close_matches(game, steam_games, n=3, cutoff=0.7)
                results[game] = [
                    match
                    for match in matches
                    if not any(string in match for string in ignore_strings)
                ]
            self.save_data_as_json(
                os.path.join(os.getcwd(), "data/close_matches.json"), results
            )
        else:
            # If file exists then load it
            results = self.load_json(close_matches_json)
        # Manually determined good keys
        good_keys = [
            "assassinscreed",
            "badnorthjotunn",
            "battlefield4",
            "beyondgoodandevil",
            "bulletgirlsphantasi",
            "daysofwar",
            "deliverusthemoonfortuna",
            "devilslayerraksasi",
            "divinityoriginalsin2",
            "dodonpachiresurrectio",
            "dragonageinquisition",
            "dungeonoftheendlesscrystal",
            "dyinglight2stayhuman",
            "empireofangelsiv",
            "europauniversalisiiicomplete",
            "evilgenius2worlddomination",
            "faiththeunholytrinity",
            "farcry2fortunes",
            "fishingnorthatlanticenhanced",
            "gujian3",
            "justcause4",
            "legendsofaria",
            "thekingoffightersxiv",
            "hotuloshuthebooksofdragon",
            "mountbladewithfiresword",
            "metroexoduspcenhanced",
            "nightoffullmoon",
            "outward",
            "pathofwuxia",
            "pathfinderkingmakerenhancedplus",
            "powerofseasons",
            "rajianancientepicenhanced",
            "riseofthetombraider20yearcelebration",
            "shadowofthetombraider",
            "skullgirls",
            "spacecolony",
            "stronghold2",
            "stronghold3gold",
            "stronghold3gold",
            "swordandfairy",
            "swordandfairyinn2",
            "taintedgrailconquest",
            "taleofimmortal",
            "taleofwuxiathepresequel",
            "thiswarofminefinal",
            "trine2completestory",
            "trüberbrook",
            "ttisleofmanrideontheedge",
            "tunshikongminglegends吞食孔明传",
            "warlockmasterofthearcanecompletecollection",
            "warmsnow",
            "thewindroad",
            "thewitcherenhanced",
            "wushuchronicles",
        ]
        df = self.mismatches.copy()
        # Here we know that the first match based on these parameters will be the correct game name through manual inspection
        # Therefore we'll resolve those game names
        for key in good_keys:
            original_key = get_close_matches(key, games, n=1, cutoff=0.4)[0]
            steam_match = results[key][0]
            app_id = self.steam_dataframe[
                self.steam_dataframe["game_normalized"] == steam_match
            ]["app_id"].tolist()[0]
            mask = df["game_normalized"] == original_key
            df.loc[mask, "app_id"] = int(app_id)
            df.loc[mask, "game_y"] = steam_match
        self.mismatches = df[df["app_id"] != 0]

    def drop_rows_being_resolved_but_still_having_zero_app_id(self):
        """
        Drop those rows where the appid couldn't be determined
        """
        dublicates = self.steam_only_games[
            self.steam_only_games.duplicated("game_normalized", keep=False)
        ]
        # get those with zero app id
        to_be_dropped = dublicates[dublicates["app_id"] == 0].index
        # drop rows by index
        self.steam_only_games.drop(to_be_dropped, inplace=True)
        self.steam_only_games.reset_index()

    def build_new_content_column(self, add_prompts: bool = False, path: str = None):
        """
        Given the game information table, adds a new column with a cleaned version of the
        detailed desctiption of the game such that it can be fed into a transformer model.

        Args:
            add_prompts (bool, optional): Whether to add prompts to the text. Defaults to False.
            path (str, optional): Where to save the dataframe. Defaults to None.
        """
        processed_descriptions = [
            self.get_content_describing_games(add_prompts, idx)
            for idx in tqdm(
                range(len(self.all_dataframe)), desc="Building detailed descriptions"
            )
        ]
        self.all_dataframe["processed_descriptions"] = processed_descriptions
        self.export_dataframe(
            self.all_dataframe,
            path,
            save_parquet=True,
        )

    @staticmethod
    def process_detailed_description(text: str) -> str:
        """Receives text and performs a set of cleaning operations to it

        Args:
            text (str): Some input text

        Returns:
            str: Cleaned version of the text
        """
        # remove html tokens
        cleantext = re.sub(CLEANR, " ", text)
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
    def add_promtps_to_text(
        text: str,
        genres: list = None,
        metacritic_score: str = None,
        controller_support: str = None,
    ) -> str:
        """Adds prompts to the text.

        Args:
            text (str): Input text
            genres (list, optional): List of genres that should be added. Defaults to None.
            metacritic_score (str, optional): Metacritic score that should be added. Defaults to None.
            controller_support (str, optional): Whether to add controller support. Defaults to None.

        Returns:
            str: Returns text with prompts.
        """
        prompts = []
        # generate prompts
        if genres is not None and genres != "nan":
            genre_prompts = [f"The genre of this game is {genre}." for genre in genres]
            [prompts.append(prompt) for prompt in genre_prompts]
        if metacritic_score is not None and metacritic_score != "nan":
            prompts.append(f"The metacritic score of the game is {metacritic_score}.")
        if controller_support is not None and controller_support != "nan":
            if controller_support == "None":
                controller_support = "no"
            prompts.append(f"The game has {controller_support} controller support.")
        # Add prompts to text
        prompt_text = " ".join(prompts)
        return text + " " + prompt_text

    def get_content_describing_games(self, add_prompts: bool = False, idx: int = 0):
        """Will format the detailed description of the game and will add prompts to the text if wanted

        Args:
            add_prompts (bool, optional): Whether to add prompts. Defaults to False.
            idx (int, optional): Idx of the game in the dataframe. Defaults to 0.

        Returns:
            _type_: _description_
        """
        description = self.process_detailed_description(
            self.all_dataframe["detailed_description"][idx]
        )
        if add_prompts == True:
            if self.all_dataframe["genre"][idx] is None:
                genres = None
            else:
                genres = list(self.all_dataframe["genre"][idx])
            # Extract data to be fed into prompts
            metacritic_score = str((self.all_dataframe["metacritic_score"][idx]))
            controller_support = str((self.all_dataframe["controller_support"][idx]))
            description = self.add_promtps_to_text(
                description, genres, metacritic_score, controller_support
            )
        return description
