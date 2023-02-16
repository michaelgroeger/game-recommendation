########################################
#  Train deep learning recommender     #
########################################
import argparse
import os
import string
from datetime import datetime

import numpy as np
import pandas as pd
import torch

torch.autograd.set_detect_anomaly(True)
import random

import wandb
from data_processors.Dataset import (
    CollaborativeFilteringDataWorker,
    HitRatioDataset,
    UserGameDataseEfficient,
)
from models.collaborative_filtering_recommender import CollabNN, DotProductBias
from tools.train import get_train_test_val_of_dataframe, train_test_validate
from tools.useful_functions import subset_users_having_at_least_n_games
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Evaluate and train algorithms")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="CollabNN",
    help="Which model architecture to train. You can set CollabNN (Neural Collaborative Filtering) or MF (Matrix Factorization).",
)
parser.add_argument(
    "-nw",
    "--nworkers",
    type=int,
    default=1,
    help="Number of workers in the Data Loader.",
)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
parser.add_argument(
    "-ed",
    "--embedding_dim",
    type=int,
    default=50,
    help="Embedding dimension of the latent user and game factors to be learnt.",
)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument(
    "-ntestu",
    "--n_test_users",
    type=int,
    default=50,
    help="How many test users to use during training for the cold start benchmark.",
)
parser.add_argument(
    "-negs",
    "--n_negative_samples",
    type=int,
    default=4,
    help="Number of negative samples per positive sample.",
)
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
parser.add_argument("-mom", "--momentum", type=float, default=0.00)
parser.add_argument("-wd", "--weight_decay", type=float, default=0.00)
parser.add_argument(
    "-mg",
    "--min_games",
    type=int,
    default=20,
    help="Minimum number of games a user need to have to be considered.",
)
parser.add_argument(
    "-mp",
    "--min_playtime",
    type=float,
    default=2.0,
    help="All games with a playtime smaller than that are set to zero.",
)
parser.add_argument(
    "-ncu",
    "--num_closest_users",
    type=int,
    default=13,
    help="Number of closest users that will be used to determine the mean user in the cold start setting.",
)
parser.add_argument("-s", "--seed", type=int, default=41)
parser.add_argument(
    "-subs",
    "--subsample_n_users",
    type=int,
    default=None,
    help="Build whole pipeline on only a subset of users",
)
parser.add_argument("--embedding_path", type=str, default=None)

parser.add_argument(
    "--checkpoint", action="store_true", help="Whether to save the best model."
)
parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
parser.add_argument(
    "--scheduling", action="store_true", help="Whether to do learning rate scheduling."
)
parser.add_argument("--no-scheduling", dest="scheduling", action="store_false")
parser.add_argument("--wandb", action="store_true")
parser.add_argument(
    "--no-wandb",
    dest="wandb",
    action="store_false",
    help="Whether to use weights&biases for experiment tracking",
)
parser.add_argument(
    "--logarithm",
    action="store_true",
    help="Whether to take the logarithm of the playtime data.",
)
parser.add_argument("--no-logarithm", dest="logarithm", action="store_false")
parser.add_argument(
    "--binarize", action="store_true", help="Whether to make playtimes binary."
)
parser.add_argument("--no-binarize", dest="binarize", action="store_false")
parser.add_argument("--feed_content_embeddings", action="store_true")
parser.add_argument(
    "--no-feed_content_embeddings", dest="feed_content_embeddings", action="store_false"
)

args = parser.parse_args()
base_path = os.path.join(os.getcwd(), "data/training_dataset")
# Load files
game_information = pd.read_parquet(
    os.path.join(
        base_path,
        "subset_game_information_5000_most_played_games_prompts=False.parq",
    )
)
user_game_matrix = pd.read_parquet(
    os.path.join(
        base_path,
        "subset_user_game_matrix_5000_most_played_games.parq",
    )
)
# Filter out all zero users
mask = (user_game_matrix == 0.0).all(axis=1)
subset_user_game_matrix = user_game_matrix[~mask]
# Take only subset of users into consideration
if args.subsample_n_users is not None:
    user_game_matrix = user_game_matrix.sample(
        args.subsample_n_users, replace=False
    ).reset_index(drop=True)


def main(
    user_game_matrix=user_game_matrix,
    game_information=game_information,
):
    # Get datetime and some randomized string for run name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    identifier = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(5)
    )
    # Subset into users fullfilling criteria
    active_users = subset_users_having_at_least_n_games(
        args.min_games, args.min_playtime, user_game_matrix
    )
    # Sample test users
    test_users = active_users.sample(
        args.n_test_users, replace=False, random_state=args.seed
    )
    # Sample non test users
    non_test_users = active_users.drop(test_users.index.tolist())
    test_users = test_users.reset_index(drop=True)
    non_test_users = non_test_users.reset_index(drop=True)
    # Instantiate wandb if it is used
    if args.wandb == True:
        wandb.init(
            name=f"{dt_string}-{args.model}-{identifier}-binary={args.binarize}-embed={args.feed_content_embeddings}",
            project="Gaming",
            entity="thesis-groeger",
        )
        wandb.config.update(vars(args))
    # Turn wide user-game matrix into short but tall dataset such that we can process it in batches
    cfd = CollaborativeFilteringDataWorker(
        non_test_users,
        game_information,
        save_all=True,
        file_storage_path=base_path,
        user_game_dataset_name=f"collaborative_filtering_dataset_subsample={args.subsample_n_users}_5000_most_played_games_min_games={args.min_games}-min_playtime{args.min_playtime}-negative_samples={args.n_negative_samples}-n_closest_users={args.num_closest_users}-seed={args.seed}-binarize={args.binarize}-logarithm={args.logarithm}.parq",
        binarize=args.binarize,
        logarithm=args.logarithm,
    )
    # Check whether this dataset already exists, if not create one
    if cfd.check_load() == False:
        # from wide to tall
        cfd.transform_dataset()
        # from app id to embedding id
        cfd.get_mappings()
        # add to dataset a embedding id based app id column
        cfd.add_mapped_app_id_columns()
    else:
        cfd.load_datasets()
    # Make train, test, val split accoring to 60-20-20 Split
    train, validate, test = get_train_test_val_of_dataframe(cfd.user_game_dataset)
    (
        train_negatives,
        validate_negatives,
        test_negatives,
    ) = get_train_test_val_of_dataframe(cfd.user_game_dataset_all_zeros)
    print(f"Excerpt of positive training data : \n{train.head()}")
    # Instantiate pytorch datasets such that they can be fed into dataloader
    train_dset = UserGameDataseEfficient(
        train, train_negatives, dataset_name="Training"
    )
    val_dset = UserGameDataseEfficient(
        validate, validate_negatives, dataset_name="Validation"
    )
    test_dset = UserGameDataseEfficient(test, test_negatives, dataset_name="Test")
    # build hit rate dataset by asking the model to rank the top game among 1000 others and
    # measure if it is under the top 10
    hit_rate_dataset = HitRatioDataset(
        test,
        cfd.user_game_dataset_all_zeros,
        dataset_name="HitRatio",
        zero_one_ratio=1000,
    )
    # instantiate data loaders
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        num_workers=args.nworkers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        num_workers=args.nworkers,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        num_workers=args.nworkers,
        shuffle=True,
    )
    # Get total number of users and games to determine the embedding length
    n_users = len(user_game_matrix.index.tolist())
    n_games = len(user_game_matrix.columns)
    print(f"Got {n_users} n_users")
    print(f"Got {n_games} n_games")
    # Assign content embeddings if selected
    if args.feed_content_embeddings == True:
        if ".npy" in args.embedding_path:
            game_content_embeddings = torch.from_numpy(np.load(args.embedding_path))
        elif ".parq" in args.embedding_path:
            temp_df = pd.read_parquet(args.embedding_path)
            game_content_embeddings = torch.from_numpy(
                temp_df[["first_axis", "second_axis", "third_axis"]].to_numpy()
            )
    elif args.feed_content_embeddings == False:
        game_content_embeddings = None
    # build models
    if args.model == "MF":
        print(f"Train Matrix Factorization with {args}")
        model = DotProductBias(
            n_users=n_users,
            n_games=n_games,
            embedding_dim=args.embedding_dim,
            idx_to_app_id=cfd.idx_to_app_id,
            app_id_to_idx=cfd.app_id_to_idx,
            reference_dataset=non_test_users,
        )
    elif args.model == "CollabNN":
        print(f"Train ColabNN with {args}")
        model = CollabNN(
            n_users=n_users,
            n_games=n_games,
            binary_classification=args.binarize,
            embedding_dim=args.embedding_dim,
            game_content_embeddings=game_content_embeddings,
            idx_to_app_id=cfd.idx_to_app_id,
            app_id_to_idx=cfd.app_id_to_idx,
            reference_dataset=non_test_users,
        )
    print(f"Model Architecture: \n{model}")
    train_test_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        hit_rate_dataset=hit_rate_dataset,
        n_epochs=args.epochs,
        lr=args.learning_rate,
        test_users=test_users,
        game_information=game_information,
        weight_decay=args.weight_decay,
        device=args.device,
        model_path=f"scripts/app/files/models",
        checkpointing=args.checkpoint,
        lr_scheduling=args.scheduling,
        binary_classification=args.binarize,
        use_wandb=args.wandb,
        identifier=identifier,
    )
    if args.wandb == True:
        wandb.finish()


if __name__ == "__main__":
    main()
