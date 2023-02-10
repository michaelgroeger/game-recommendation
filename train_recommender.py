########################################
#  Train deep learning recommender     #
########################################
import argparse
import os
import string
from datetime import datetime

import pandas as pd
import torch

torch.autograd.set_detect_anomaly(True)
import random

from data_processors.Dataset import (
    CollaborativeFilteringDataWorker,
    HitRatioDataset,
    UserGameDataseEfficient,
)
from models.collaborative_filtering_recommender import CollabNN, DotProductBias
from tools.train import get_train_test_val_of_dataframe, train_test_validate
from tools.useful_functions import sample_users_having_at_least_n_games
from torch.utils.data import DataLoader

import wandb

parser = argparse.ArgumentParser(description="Evaluate and train algorithms")
parser.add_argument("-m", "--model", type=str, default="CollabNN")
parser.add_argument("-mo", "--mode", type=str, default="local")
parser.add_argument("-nw", "--nworkers", type=int)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
parser.add_argument("-ed", "--embedding_dim", type=int, default=50)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-ntestu", "--n_test_users", type=int, default=50)
parser.add_argument("-negs", "--n_negative_samples", type=int, default=4)
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
parser.add_argument("-mom", "--momentum", type=float, default=0.00)
parser.add_argument("-wd", "--weight_decay", type=float, default=0.00)
parser.add_argument("-mg", "--min_games", type=int, default=20)
parser.add_argument("-mp", "--min_playtime", type=float, default=2.0)
parser.add_argument("-ncu", "--num_closest_users", type=int, default=5)
parser.add_argument("-s", "--seed", type=int, default=41)
parser.add_argument("-subs", "--subsample_n_users", type=int, default=None)

parser.add_argument("--checkpoint", action="store_true")
parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
parser.add_argument("--scheduling", action="store_true")
parser.add_argument("--no-scheduling", dest="scheduling", action="store_false")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--no-wandb", dest="wandb", action="store_false")
parser.add_argument("--logarithm", action="store_true")
parser.add_argument("--no-logarithm", dest="logarithm", action="store_false")
parser.add_argument("--binarize", action="store_true")
parser.add_argument("--no-binarize", dest="binarize", action="store_false")

args = parser.parse_args()

if args.mode == "local":
    base_path = (
        "/Users/michaelgroeger/workspace/data-science-project/data/training_dataset"
    )
    logger_dir = "/Users/michaelgroeger/workspace/data-science-project/logs"
elif args.mode == "Callisto":
    base_path = "/shared-network/mgroeger/datasets/project/data/training_dataset"
    logger_dir = "/shared-network/mgroeger/logs/project"
else:
    base_path = "/mnt/qb/work/akata/mgroeger27/datasets/project/data"
# Load files
game_information = pd.read_parquet(
    os.path.join(
        os.getcwd(),
        "data/training_dataset/subset_game_information_5000_most_played_games_prompts=False.parq",
    )
)
user_game_matrix = pd.read_parquet(
    os.path.join(
        os.getcwd(),
        "data/training_dataset/subset_user_game_matrix_5000_most_played_games.parq",
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
    active_users = sample_users_having_at_least_n_games(
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
            name=f"{dt_string}-{args.model}-{identifier}-binary={args.binarize}",
            project="Gaming",
            entity="thesis-groeger",
            dir=logger_dir,
        )
        wandb.config.update(vars(args))
    # Turn wide user-game matrix into short but tall dataset such that we can process it in batches
    cfd = CollaborativeFilteringDataWorker(
        non_test_users,
        game_information,
        save_all=True,
        file_storage_path=base_path,
        user_game_dataset_name=f"collaborative_filtering_dataset_subsample={args.subsample_n_users}_5000_most_played_games_min_games={args.min_games}-min_playtime{args.min_playtime}-negative_samples={args.n_negative_samples}-n_closest_users={args.num_closest_users}-seed={args.seed}.parq",
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
    # Instantiate pytorch datasets such that they can be fed into dataloader
    train_dset = UserGameDataseEfficient(
        train, train_negatives, dataset_name="Training"
    )
    val_dset = UserGameDataseEfficient(
        validate, validate_negatives, dataset_name="Validation"
    )
    test_dset = UserGameDataseEfficient(test, test_negatives, dataset_name="Test")
    # build hit rate dataset
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
    # build models
    if args.model == "MF":
        print(f"Train Matrix Factorization with {args}")
        model = DotProductBias(
            n_users=n_users,
            n_games=n_games,
            embedding_dim=args.embedding_dim,
            idx_to_app_id=cfd.idx_to_app_id,
            app_id_to_idx=cfd.app_id_to_idx,
            reference_dataset=user_game_matrix,
        )
    elif args.model == "CollabNN":
        print(f"Train ColabNN with")
        model = CollabNN(
            n_users=n_users,
            n_games=n_games,
            binary_classification=args.binarize,
            embedding_dim=args.embedding_dim,
            idx_to_app_id=cfd.idx_to_app_id,
            app_id_to_idx=cfd.app_id_to_idx,
            reference_dataset=None,
        )
    print(model)
    # Attach more params to model such that they can be saved alongside it
    model.min_games = args.min_games
    model.min_playtime = args.min_playtime
    model.n_negative_samples = args.n_negative_samples
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
        model_path=f"/shared-network/mgroeger/models/project/{args.model}",
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
