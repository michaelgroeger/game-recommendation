import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scrapers.NvidiaScraper import NvidiaScraper
from scrapers.SteamScraper import SteamScraper
from sklearn.metrics.pairwise import cosine_similarity

#####################################################################################################################
# A collection of different functions that don't fit into a specific niche but are commonly used in different tasks #
#####################################################################################################################


def get_Steam_games(url: str, path_to_steam_games: str = None) -> list:
    """Will either load json from local or download again the data, save and return it

    Args:
        url (str): Url to GET data from
        path_to_steam_games (str): local path to predownloaded data if exists

    Returns:
        list:
    """
    steam_1 = SteamScraper(url=url)
    if path_to_steam_games != None and Path(path_to_steam_games).is_file():
        games = steam_1.get_url_content()
        steam_1.save_data_as_json(
            path_to_steam_games,
            games,
        )
        return games
    else:
        return steam_1.get_url_content()


def load_config(path: str) -> dict:
    """Loads config file from yaml

    Args:
        path (str): path to config.yaml

    Returns:
        dict: Nested Dict holding configurations
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_Nvidia_games(
    url: str,
    classname: str,
    get_first_n_games: int = None,
    path_to_nvidia_games: str = None,
) -> list:
    """
    Will scrape data if path_to_nvidia_games does not exist or load data if it does.

    Args:
        url (str): url where games are listed
        classname (str): javascript token which holds game information
        get_first_n_games (int, optional): Whether to limit the scraping to the first n games. Defaults to None.
        path_to_nvidia_games (str): Path where priorly scraped games are saved.

    Returns:
        list: game names
    """
    if (
        path_to_nvidia_games != None
        and not Path(os.path.join(os.getcwd(), path_to_nvidia_games)).is_file()
    ):
        # instantiate scraper
        nvidia_1 = NvidiaScraper(
            url=url, classname=classname, get_first_n_games=get_first_n_games
        )
        nvidia_1.build_chrome_driver()
        # collect
        nvidia_1.get_games()
        # save
        nvidia_1.pickle_data(path_to_nvidia_games, nvidia_1.scraped_content)
        return nvidia_1.scraped_content
    else:
        nvidia_1 = NvidiaScraper(
            url=url, classname=classname, get_first_n_games=get_first_n_games
        )
        nvidia_1.scraped_content = nvidia_1.unpickle_data(
            os.path.join(os.getcwd(), path_to_nvidia_games)
        )
        if get_first_n_games != None:
            nvidia_1.scraped_content = nvidia_1.scraped_content[:get_first_n_games]
        return nvidia_1.scraped_content


def get_user_games(
    user_id, user_game_matrix, only_played=True, must_be_present_in_dataset=True
):
    """Given a user id will return the played games of a user. Optionally filter for games that were actually played and
    whether they are present in the current dataset.


    Args:
        user_id (str): Unique steam user id
        user_game_matrix (pd.DataFrame): pandas dataframe of the user-game interactions
        only_played (bool, optional): Whether to filter for games that were actually played. Defaults to True.
        must_be_present_in_dataset (bool, optional): Whether to filter out games that are not in the user item matrix. Defaults to True.

    Returns:
        list, list: Outputs lists of game names, appids, playtimes
    """
    steam_scraper = SteamScraper()
    query_output = steam_scraper.query_player_and_games(user_id=user_id)
    game_name = []
    app_id = []
    playtimes = []
    # collect games and apps that were played)
    for game in query_output["gamesList"]["games"]["game"]:
        if only_played == True:
            if "hoursOnRecord" in game.keys():
                game_name.append(game["name"])
                app_id.append(game["appID"])
                playtimes.append(game["hoursOnRecord"])
        else:
            game_name.append(game["name"])
            app_id.append(game["appID"])
    if must_be_present_in_dataset == True:
        app_id_present = [i for i in app_id if i in user_game_matrix.columns]
        game_name_present = [game_name[app_id.index(i)] for i in app_id_present]
        if only_played == True:
            playtimes_present = [playtimes[app_id.index(i)] for i in app_id_present]
            return game_name_present, app_id_present, playtimes_present
        return game_name_present, app_id_present
    else:
        return game_name, app_id


def get_sim_score(
    embedding_1, embedding_2, cosine_fct=torch.nn.CosineSimilarity(dim=1)
):
    """Get similarity between two tensors based on cosine similarity

    Args:
        embedding_1 (torch.Tensor): Embedding Tensor
        embedding_2 (torch.Tensor): Embedding Tensor
    Returns:
        float, similarity score based on cosine similarity
    """
    with torch.no_grad():
        return torch.sum(cosine_fct(embedding_1, embedding_2))


def get_closest_vectors(vector, matrix, top_k=10):
    """Given a matrix and a vector. Returns the top k closest vektors in the matrix based in cosine similarity with input vector.

    Args:
        vector (np.Array): Input Vector
        matrix (np.Array): Matrix holding other vectors we want to get the similarity to the input vector to.
        top_k (int, optional): Return top k closest vectors. Defaults to 10.

    Returns:
        np.array, np.arry, np.array: Similarities of all vectors, indices of the closest ones, similarities of the clostest ones
    """
    similarities = cosine_similarity(X=vector, Y=matrix)[0]
    max_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_k_sim = similarities[max_indices]
    return similarities, max_indices, top_k_sim


def subset_users_having_at_least_n_games(
    n_games: int, min_playtime: float, user_game_matrix: pd.DataFrame
):
    """Will set all games with a playtime smaller then min playtime to zero and will only consider those users that have at least n_games.

    Args:
        n_games (int): Number of games a user needs to have to make it into the final dataset.
        min_playtime (float): Games with a lower playtime will be set to zero.
        user_game_matrix (pd.DataFrame): Dataframe with user game interaction data.

    Returns:
        pd.DataFrame: User-game matrix subset by parameters
    """
    df_subset = user_game_matrix.copy()
    df_subset[df_subset < min_playtime] = 0.0
    df_subset = df_subset.loc[(df_subset != 0.0).sum(axis=1) >= n_games].copy()
    return df_subset


def create_dummy_dataset(user_game_matrix, n_unique_users, repeat_user_n_times):
    """Creates an artificial dataset

    Args:
        user_game_matrix (pd.DataFrame): Body for dummy dataset
        n_unique_users (int): Number of unique users in artificial dataset
        repeat_user_n_times (int): how often a user should be repeated

    Returns:
        pd.DataFrame: Dummy dataset
    """
    # Create dummy matrix and set all to zero
    dummy_matrix = user_game_matrix.copy()
    dummy_matrix[dummy_matrix > 0.0] = 0.0
    counter = 0
    for _ in range(n_unique_users):
        # sample uniformly 50 playtimes
        sampl = np.random.uniform(low=0.01, high=200, size=(50,))
        # sample random columns that are games
        cols = random.sample(user_game_matrix.columns.tolist(), 50)
        for i in range(repeat_user_n_times):
            # fill the sampled games with the sampled playtimes n times
            dummy_matrix.iloc[counter, :][cols] = sampl
            counter += 1
    # drop allzero rows
    mask = (dummy_matrix == 0.0).all(axis=1)
    dummy_matrix = dummy_matrix[~mask]
    return dummy_matrix
