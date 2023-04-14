##################################################################################################################################################
# Functions and classes for the pytorch training loop and for processing data to arrive at a new pandas dataframe to be used in downstream tasks #
##################################################################################################################################################
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_processors.AllDataProcessor import AllDataProcessor
from torch.utils.data import Dataset
from tqdm import tqdm


class UserGameDataseEfficient(Dataset):
    """User Game dataset for pytorch training"""

    def __init__(
        self,
        user_game_dataset: pd.DataFrame,
        user_game_dataset_all_zeros: pd.DataFrame,
        dataset_name: str,
        zero_one_ratio: int = 2,
    ):
        """Initializes instance of class UserGameDataset.

        Args:
            user_game_dataset (pd.DataFrame): Dataframe of containing userids, appid and playtime, where playtime > 0. Output of the CollaborativeFilteringDataWorker.
            user_game_dataset_all_zeros (pd.DataFrame): Dataframe of containing userids, appid and playtime, where playtime = 0. Output of the CollaborativeFilteringDataWorker.
            dataset_name (str): Name of the dataset so when we resample the negatives we can display which dataset that is.
            zero_one_ratio (int, optional): How many negative per positive should be sampled. Defaults to 2.
        """
        super(UserGameDataseEfficient, self).__init__()
        self.user_game_dataset = user_game_dataset
        self.user_game_dataset_all_zeros = user_game_dataset_all_zeros
        self.dataset_name = dataset_name
        self.zero_one_ratio = zero_one_ratio
        # Call add negative to build full dataset
        self.add_negatives()

    def __len__(self):
        # 0 = user ids because self.epoch_dataset is tuple of three tensors of same length
        return len(self.epoch_dataset[0])

    def add_negatives(self):
        # Report to terminal which dataset is being build
        print(
            f"Building new version of dataset {self.dataset_name} by providing new negative samples for next run."
        )
        # Get all positives
        app_id_positive = pd.Series(self.user_game_dataset["app_id_mapped"])
        user_id_positive = pd.Series(self.user_game_dataset["user_id"])
        playtime_positive = pd.Series(self.user_game_dataset["playtime"])
        # Get sample enough negatives such that we have zero_one_ratio per positive
        negative_data = self.user_game_dataset_all_zeros.sample(
            len(self.user_game_dataset) * self.zero_one_ratio
        )
        # Concat positives and negatives
        user_ids = pd.concat(
            [user_id_positive, negative_data["user_id"]], axis=0
        ).reset_index(drop=True)
        app_ids = pd.concat(
            [app_id_positive, negative_data["app_id_mapped"]], axis=0
        ).reset_index(drop=True)
        playtimes = pd.concat(
            [playtime_positive, negative_data["playtime"]], axis=0
        ).reset_index(drop=True)
        user_ids = user_ids.values
        app_ids = app_ids.values
        playtimes = playtimes.values
        # Put into dataframe and set values
        self.epoch_dataset = pd.DataFrame(
            {"user_id": user_ids, "app_id_mapped": app_ids, "playtime": playtimes}
        )
        self.epoch_dataset = self.epoch_dataset.astype(
            {"user_id": int, "app_id_mapped": int, "playtime": float}
        )
        # Shuffle dataset
        self.epoch_dataset = self.epoch_dataset.sample(frac=1).reset_index(drop=True)
        self.epoch_dataset = (
            torch.LongTensor(self.epoch_dataset["user_id"].values),
            torch.LongTensor(self.epoch_dataset["app_id_mapped"].values),
            torch.DoubleTensor(self.epoch_dataset["playtime"].values),
        )
        assert (
            len(self.epoch_dataset[0])
            == len(self.epoch_dataset[1])
            == len(self.epoch_dataset[2])
        ), f"Something went wrong, length of user ids, appids, playtimes differ but should be the same"

    def __getitem__(self, idx):
        user_ids = self.epoch_dataset[0][idx]
        app_ids = self.epoch_dataset[1][idx]
        playtime = self.epoch_dataset[2][idx]
        return user_ids, app_ids, playtime


class HitRatioDataset(Dataset):
    """
    Creates Dataset that is used to calculate the HitRatio metric during training.
    """

    def __init__(
        self,
        user_game_dataset: pd.DataFrame,
        user_game_dataset_all_zeros: pd.DataFrame,
        dataset_name: str,
        zero_one_ratio: int = 1000,
        sample_n_unique_users: int = 100,
    ):
        """Initializes instance of class HitRatio.

        Args:
            user_game_dataset (pd.DataFrame): Dataframe of containing userids, appid and playtime, where playtime > 0. Output of the CollaborativeFilteringDataWorker.
            user_game_dataset_all_zeros (pd.DataFrame): Dataframe of containing userids, appid and playtime, where playtime = 0. Output of the CollaborativeFilteringDataWorker.
            dataset_name (str): Name of the dataset so when we resample the negatives we can display which dataset that is.
            zero_one_ratio (int, optional): How many negative per positive should be sampled. Defaults to 1000.
            sample_n_unique_users (int, optional): Based on how many unique useres the hit ratio will be calculates. Defaults to 100.
        """
        super(HitRatioDataset, self).__init__()
        self.user_game_dataset = user_game_dataset
        self.user_game_dataset_all_zeros = user_game_dataset_all_zeros
        self.dataset_name = dataset_name
        self.zero_one_ratio = zero_one_ratio
        self.sample_n_unique_users = sample_n_unique_users
        # Call add negative to build full dataset
        self.add_negatives()

    def __len__(self):
        return len(self.epoch_dataset)

    def sample_unique_positves(self):
        """
        sample n unique users from dataset and take one positive for each user
        """
        # Sample users
        users = np.random.choice(
            self.user_game_dataset["user_id"].unique(),
            (self.sample_n_unique_users),
            replace=False,
        )
        # Sample one positive per user
        self.hit_ratio_positives = pd.DataFrame()
        for user in users:
            self.hit_ratio_positives = pd.concat(
                [
                    self.hit_ratio_positives,
                    self.user_game_dataset[
                        self.user_game_dataset["user_id"] == user
                    ].sample(1),
                ]
            )
        self.hit_ratio_positives

    def add_negatives(self):
        # Takes one positive instance and adds zero_one_ratio negatives to it and exports it to a dataframe
        print(
            f"Building new version of dataset {self.dataset_name} by providing new negative samples for next run."
        )
        self.sample_unique_positves()
        app_ids = pd.Series(self.hit_ratio_positives["app_id_mapped"])
        user_ids = pd.Series(self.hit_ratio_positives["user_id"])
        playtimes = pd.Series(self.hit_ratio_positives["playtime"])
        # Add negatives per user
        for user in user_ids.values:
            # Sample zero_one_ratio negatives and add to positives
            negative_data = self.user_game_dataset_all_zeros.loc[
                self.user_game_dataset_all_zeros["user_id"] == user
            ].sample(self.zero_one_ratio)
            # Separate user, app ids and playtime
            user_ids = pd.concat(
                [user_ids, negative_data["user_id"]], axis=0
            ).reset_index(drop=True)
            app_ids = pd.concat(
                [app_ids, negative_data["app_id_mapped"]], axis=0
            ).reset_index(drop=True)
            playtimes = pd.concat(
                [playtimes, negative_data["playtime"]], axis=0
            ).reset_index(drop=True)
        # Export all to dataframe
        self.epoch_dataset = pd.DataFrame(
            {"user_id": user_ids, "app_id_mapped": app_ids, "playtime": playtimes}
        )
        # Set types
        self.epoch_dataset = self.epoch_dataset.astype(
            {"user_id": int, "app_id_mapped": int, "playtime": float}
        )

    def __getitem__(self, user_id):
        # Query 1 batch per user that is one positive and all their negatives. Transfer data to Tensors
        user_ids_tensor = torch.LongTensor(
            self.epoch_dataset[self.epoch_dataset["user_id"] == user_id][
                "user_id"
            ].values
        )
        app_ids_tensor = torch.LongTensor(
            self.epoch_dataset[self.epoch_dataset["user_id"] == user_id][
                "app_id_mapped"
            ].values
        )
        playtime_tensor = torch.DoubleTensor(
            self.epoch_dataset[self.epoch_dataset["user_id"] == user_id][
                "playtime"
            ].values
        )
        return user_ids_tensor, app_ids_tensor, playtime_tensor


class CollaborativeFilteringDataWorker:
    def __init__(
        self,
        user_game_matrix: pd.DataFrame,
        game_information: pd.DataFrame,
        save_all: bool = True,
        file_storage_path: str = "/".join(os.getcwd().split("/")[:-2]),
        user_game_dataset_name: str = None,
        seed: int = 42,
        binarize: bool = False,
        logarithm: bool = False,
        build_all_zero_dataset: bool = True,
    ):
        """
        DataWorker that acts on the User-Game matrix and exports a Dataframe that can be fed into the UserGameDataset and HitRatioDataset classes. Essentially collapses the
        Wide user-item matrix into a four column long DataFrame.

        Args:
            user_game_matrix (pd.DataFrame): User-Game Matrix of shape n_users x n_games
            game_information (pd.DataFrame): Dataframe holding game side-features
            save_all (bool, optional): Whether to export the Dataframe. Defaults to True.
            file_storage_path (_type_, optional): Basepath where to save the Dataframe. Defaults to "/".join(os.getcwd().split("/")[:-2]).
            user_game_dataset_name (str, optional): Name of the Dataframe such that it can be loaded later again. Defaults to None.
            seed (int, optional): Seed for sampling operations. Defaults to 42.
            binarize (bool, optional): Whether to binarize dataset. Defaults to False.
            logarithm (bool, optional): Whether to take the log of the dataset. Defaults to False.
            build_all_zero_dataset (bool, optional): Whether to build a dataset of all zeros. Defaults to True. If False will build dataset of only played games.
        """
        self.game_information = game_information
        self.user_game_matrix = user_game_matrix
        self.save_all = save_all
        self.file_storage_path = file_storage_path
        self.user_game_dataset_name = user_game_dataset_name
        self.seed = seed
        self.binarize = binarize
        self.logarithm = logarithm
        self.build_all_zero_dataset = build_all_zero_dataset

    def check_load(self):
        """Checks if a dataset with the same name exists already. If yes then loads it.

        Returns:
            Bool: True: found and load dataset, False: Did't find any dataset.
        """
        # Check of positives and negatives exist.
        previous_positives = Path(
            os.path.join(self.file_storage_path, self.user_game_dataset_name)
        )
        previous_negatives = Path(
            os.path.join(
                self.file_storage_path, "all_negatives_" + self.user_game_dataset_name
            )
        )
        # Load and return True
        if previous_positives.is_file() & previous_negatives.is_file():
            self.load_datasets()
            return True
        return False

    def load_datasets(self):
        """
        Loads datasets
        """
        # Load positives
        self.user_game_dataset = pd.read_parquet(
            os.path.join(self.file_storage_path, self.user_game_dataset_name),
        )
        # Load negatives
        self.user_game_dataset_all_zeros = pd.read_parquet(
            os.path.join(
                self.file_storage_path, "all_negatives_" + self.user_game_dataset_name
            )
        )
        self.get_mappings()
        print("Loaded datasets and created mappings")

    def get_mappings(self):
        # We need to map the user indices and appids to 0-n because we will use their id
        # to access their embedding position to later come back to the original appid we need those mappings
        self.idx_to_app_id = {
            idx: self.user_game_matrix.columns[idx]
            for idx in range(len(self.user_game_matrix.columns))
        }
        self.app_id_to_idx = {v: k for k, v in self.idx_to_app_id.items()}

    def transform_dataset(self):
        """
        Core function turing wide user-game matrix to tall four column dataframe
        """
        user_ids = []
        app_ids = []
        playtime = []
        user_ids_negatives = []
        app_ids_negatives = []
        playtime_negatives = []
        # Collect data per row
        progress_bar = tqdm(
            range(len(self.user_game_matrix)), desc="Rebuilding dataframe"
        )
        for index in progress_bar:
            # Get positives from row
            row = self.user_game_matrix.iloc[index, :].copy()
            relevant_cols = row[row != 0.0]
            # Take logarithm
            if self.logarithm == True:
                # Scale all by 1 because we might have values < 1 which then would be negative
                relevant_cols = relevant_cols + 1.0
                relevant_cols = np.log(relevant_cols)
            if self.build_all_zero_dataset == True:
                # get all negatives from row
                negatives = row[row == 0.0]
                # Add negatives to list
                playtime_negatives.append([float(value) for value in negatives])
                app_ids_negatives.append(negatives.index.to_list())
                user_ids_negatives.append([index for i in range(len(negatives))])
            # Add positives to list
            playtime.append([float(value) for value in relevant_cols])
            app_ids.append(relevant_cols.index.to_list())
            user_ids.append([index for i in range(len(relevant_cols))])
        # Flatten list of lists of positives
        user_ids_flat = [item for sublist in user_ids for item in sublist]
        app_ids_flat = [item for sublist in app_ids for item in sublist]
        playtime_flat = [item for sublist in playtime for item in sublist]
        if self.build_all_zero_dataset == True:
            # Flatten list of lists of negatives
            user_ids_negatives_flat = [
                item for sublist in user_ids_negatives for item in sublist
            ]
            app_ids_negatives_flat = [
                item for sublist in app_ids_negatives for item in sublist
            ]
            playtime_negatives_flat = [
                item for sublist in playtime_negatives for item in sublist
            ]
            # Put negatives into dataframe
            self.user_game_dataset_all_zeros = pd.DataFrame(
                {
                    "user_id": user_ids_negatives_flat,
                    "app_id": app_ids_negatives_flat,
                    "playtime": playtime_negatives_flat,
                }
            )
        # Put positives into dataframe
        self.user_game_dataset = pd.DataFrame(
            {
                "user_id": user_ids_flat,
                "app_id": app_ids_flat,
                "playtime": playtime_flat,
            }
        )
        # Binarize all positives if it is demanded
        if self.binarize == True:
            self.user_game_dataset["playtime"][
                self.user_game_dataset["playtime"] > 0
            ] = 1.0
        # Convert column names to string
        self.user_game_dataset.columns = self.user_game_dataset.columns.astype(str)

    def add_mapped_app_id_columns(self):
        """
        Since the appids of the output of tranform_data will be the original ones, we need to add the mapped id to the dataframes such that we can adress the correct embedding in the model.
        """
        # Add column to positives
        self.user_game_dataset["app_id_mapped"] = [
            int(self.app_id_to_idx[i])
            for i in tqdm(
                self.user_game_dataset["app_id"], desc="Adding mapping to positives"
            )
        ]
        if self.build_all_zero_dataset == True:
            # Add column to negatives
            self.user_game_dataset_all_zeros["app_id_mapped"] = [
                int(self.app_id_to_idx[i])
                for i in tqdm(
                    self.user_game_dataset_all_zeros["app_id"],
                    desc="Adding mapping to negatives",
                )
            ]
        if self.save_all == True:
            # Export datasets
            self.user_game_dataset.to_parquet(
                os.path.join(self.file_storage_path, self.user_game_dataset_name),
                index=False,
            )
            if self.build_all_zero_dataset == True:
                self.user_game_dataset_all_zeros.to_parquet(
                    os.path.join(
                        self.file_storage_path,
                        "all_negatives_" + self.user_game_dataset_name,
                    ),
                    index=False,
                )


##############################################################################################
# Functions to generate the core input for the model trainings which are based on two files: #
#  - data/raw/processed_game_information.parq -> Containing the side features            #
#  - data/raw/user_game_matrix_float.parq -> Raw user game matrix                        #
#  Outputs:                                                                                  #
#  - subset_game_information                                                                 #
#  - subset_user_game_matrix                                                                 #
#  - Both based on the top-n most played games set by the user in the function:              #
#  build_training_dataset.py                                                                 #
##############################################################################################


def return_harmonized_user_game_matrix_and_processed_game_information(
    game_information: pd.DataFrame, user_game_matrix: pd.DataFrame
) -> pd.DataFrame:
    """ "As an artifact from the data acquisiton phase the game information table has less games  than the user-game matrix. Here we need to fix this.

    Args:
        game_information (pd.DataFrame): Dataframe containing side features of the games
        user_game_matrix (pd.DataFrame): Dataframe of the user-game playtimes

    Returns:
        pd.DataFrame: Harmonized versions of both datasets.
    """
    # get mismatch
    print(
        f"Game information has {user_game_matrix.shape[1] - game_information.shape[0]} fewer games than user-item matrix. Will harmonize them."
    )
    mismatch_games = list(
        set(user_game_matrix.columns) - set(game_information["appid"].tolist())
    )
    # drop columns that are not in both tables
    user_game_matrix_harmonized = user_game_matrix.drop(mismatch_games, axis=1)
    return game_information, user_game_matrix_harmonized


def get_n_highest_played_games(
    game_information: pd.DataFrame,
    user_game_matrix: pd.DataFrame,
    n_highest: int,
    return_data: bool = True,
) -> pd.DataFrame:
    """Based on the user input of will return subset of game information and the user_game matrix based on the n_highest most played games

    Args:
        game_information (pd.DataFrame): Dataframe containing side features of the games
        user_game_matrix (pd.DataFrame): Dataframe of the user-game playtimes
        n_highest (int): Number determining the top_n most played games to consider.
        return_data (bool, optional): Whether to return the new dataframes. Defaults to True.

    Returns:
        pd.DataFrame: Subset versions of both datasets.
    """
    print(
        f"Building subsets of game information and user-item matrix based on {n_highest} most commonly played games."
    )
    # Sum over all rows to get the total absolute playtime
    df_total_game_time = user_game_matrix.iloc[:, :].sum(axis=0).to_frame().T
    # sort columns by playtime
    df_total_game_time.sort_values(
        by=0,
        axis=1,
        ascending=False,
        inplace=True,
        kind="quicksort",
        na_position="last",
    )
    # Subset the game information into only those games which are present in total game time and amongst n_highest
    subset_game_information = game_information.loc[
        game_information["appid"].isin(
            list(df_total_game_time.iloc[:, :n_highest].columns)
        )
    ]
    # Subset user game matrix, select all games which are also in game information
    subset_user_game_matrix = user_game_matrix[
        subset_game_information["appid"].tolist()
    ]
    # Make sure both have the same games
    assert set(subset_user_game_matrix.columns) == set(
        subset_game_information["appid"]
    ), f"Subset User item matrix and subset game information don't contain same games"
    if return_data == True:
        return subset_game_information, subset_user_game_matrix
    else:
        return subset_game_information["appid"]


mapping = {
    "Action": "Action",
    "Simulation": "Simulation",
    "Casual": "Casual",
    "Adventure": "Adventure",
    "Strategy": "Strategy",
    "Free to Play": "Free to Play",
    "Indie": "Indie",
    "Racing": "Racing",
    "Utilities": "Utilities",
    "Unknown": "Unknown",
    "RPG": "RPG",
    "Massively Multiplayer": "Massively Multiplayer",
    "Sports": "Sports",
    "Early Access": "Early Access",
    "Utilitaires": "Utilities",
    "Экшены": "Action",
    "Приключенческие игры": "Adventure",
    "Инди": "Indie",
    "Design & Illustration": "Utilities",
    "Rollenspiel": "RPG",
    "Software Training": "Utilities",
    "Gelegenheitsspiele": "Casual",
    "アクション": "Action",
    "Animation & Modeling": "Utilities",
    "Violent": "Action",
    "Nudity": "Action",
    "Aktion": "Action",
    "Ação": "Action",
    "Sexual Content": "Adult Content",
    "Gore": "Adult Content",
    "冒险": "Action",
    "Video Production": "Utilities",
    "动作": "Action",
    "模擬": "Simulation",
    "Audio Production": "Utilities",
    "Web Publishing": "Utilities",
    "Azione": "Action",
    "Game Development": "Utilities",
    "Movie": "Utilities",
    "動作": "Action",
    "Казуальные игры": "Casual",
    "Education": "Utilities",
    "Documentary": "Utilities",
    "Short": "Utilities",
    "Photo Editing": "Utilities",
    "Accounting": "Utilities",
    "獨立製作": "Indie",
}


def map_genres(
    game_informations: pd.DataFrame, mapping: dict = mapping
) -> pd.DataFrame:
    """There are a lot of noisy and dublicated genres in the Steam data. We map them here to some reasonable ones

    Args:
        game_information (pd.DataFrame): Dataframe containing side features of the games
        mapping (dict, optional): Dictionary holding mapped generes. Defaults to mapping which is defined above the source code for that function.

    Returns:
        pd.DataFrame: Subset versions of both datasets.
    """
    # First extract the first genre in a list of genres for each game
    genres = []
    for i in range(len(game_informations)):
        # Get genre
        genre = game_informations.iloc[i, :]["genre"]
        # Turn None to Unknown
        if genre is None:
            genre = "Unknown"
            genres.append(genre)
        else:
            genres.append(genre[0])
    # Create new column
    game_informations["single_genre"] = genres
    # Fill with mapped versions of the first extracted genre per game
    game_informations["single_genre"] = game_informations["single_genre"].map(mapping)
    return game_informations


def build_training_datasets(
    n_highest: int,
    save: bool = True,
    add_prompts: bool = True,
    path_to_raw_game_information: str = "data/raw/processed_game_information.parq",
    path_to_raw_user_game_matrix: str = "data/raw/user_game_matrix_float.parq",
    harmonize_genres: bool = True,
):
    """Will build a subset of the entire data based on the top n_most played games

    Args:
        n_highest (int): Consider n most played games
        save (bool, optional): Whether to save the results. Defaults to True.
        add_prompts (bool, optional): Whether to add prompts to the detailed descriptions of the games. Defaults to True.
        path_to_raw_game_information (str, optional): Path to raw game information. Defaults to "data/raw/processed_game_information.parq".
        path_to_raw_user_game_matrix (str, optional): Path to raw user game matrix. Defaults to "data/raw/user_game_matrix_float.parq".
        harmonize_genres (bool, optional): Whether to map genres to some known ones. Defaults to True.

    Returns:
        _type_: _description_
    """
    base_path = os.getcwd()
    # Load raw data
    user_game_matrix = pd.read_parquet(
        os.path.join(base_path, path_to_raw_user_game_matrix)
    )
    game_information = pd.read_parquet(
        os.path.join(base_path, path_to_raw_game_information)
    )
    print(f"Sucessfully loaded raw data")
    # Build path for output
    training_dataset_path = Path(os.path.join(base_path, "data/training_dataset"))
    if not training_dataset_path.is_dir():
        os.mkdir(training_dataset_path)
        print(f"Created folder")
    # Make sure both conatain the same games
    (
        game_information,
        user_game_matrix_harmonized,
    ) = return_harmonized_user_game_matrix_and_processed_game_information(
        game_information, user_game_matrix
    )
    print(f"Harmonized data")
    # Subset data to n_highest
    subset_game_information, subset_user_game_matrix = get_n_highest_played_games(
        game_information, user_game_matrix_harmonized, n_highest=n_highest
    )
    subset_game_information = subset_game_information.reset_index(drop=True)
    print(f"Got n_highest games for game informations and user-game matrix")
    # Map genres based on mapping
    if harmonize_genres == True:
        subset_game_information = map_genres(subset_game_information)
    # Generate the cleaned processed descriptions
    # We need to process the detailed description. Instantiate a processor and process
    data_processor = AllDataProcessor(all_dataframe=subset_game_information)
    subset_game_information_path = os.path.join(
        training_dataset_path,
        f"subset_game_information_{n_highest}_most_played_games_prompts={add_prompts}.parq",
    )
    data_processor.build_new_content_column(
        add_prompts=add_prompts,
        path=subset_game_information_path,
    )
    # Drop users that don't have any playtime
    mask = (user_game_matrix_harmonized == 0.0).all(axis=1)
    user_game_matrix_harmonized = user_game_matrix_harmonized[~mask]
    # Export results
    if save == True:
        subset_user_game_matrix_path = os.path.join(
            training_dataset_path,
            f"subset_user_game_matrix_{n_highest}_most_played_games.parq",
        )
        subset_user_game_matrix.to_parquet(subset_user_game_matrix_path)
        print(
            f"Saved data: \nsubset_user_game_matrix to {subset_user_game_matrix_path}\nsubset_game_information_path to {subset_game_information_path}"
        )
    print("Done")
    return subset_game_information, subset_user_game_matrix
