######################################################
#  Evaluate recommenders in a cold start setting     #
######################################################
import argparse
import os

import numpy as np
import pandas as pd
import torch
from models.collaborative_filtering_recommender import CollabNNInference
from tools.train import evaluate_recommender
from tools.useful_functions import sample_users_having_at_least_n_games

parser = argparse.ArgumentParser(description="Evaluate and train algorithms")
parser.add_argument(
    "-mpath",
    "--model_path",
    type=str,
    default=None,
    description="Path to trained model you want to evaluate",
)
parser.add_argument(
    "-s", "--seed", type=int, default=41, description="Seed for sampling operations"
)
# Occlusion means how many % of the games we try to reconstruct
parser.add_argument(
    "-o",
    "--occlusion",
    type=float,
    default=0.3,
    description="Level of occlusion of the test user",
)
parser.add_argument(
    "-mg",
    "--min_games",
    type=int,
    default=20,
    description="Minimum amount of games a user need to have to be considered",
)
parser.add_argument(
    "-nu", "--n_test_users", type=int, default=20, description="Amount of test users"
)
parser.add_argument(
    "-ncu",
    "--n_closest_users",
    type=int,
    default=13,
    description="For the deep and naive recommender ranking stage, how many users should be used to built the mean embedding",
)
parser.add_argument(
    "-ncand",
    "--n_candidate_games",
    type=int,
    default=120,
    description="For the deep recommender retrival stage - How many candidate games to retrieve",
)
parser.add_argument(
    "-mp",
    "--min_playtime",
    type=float,
    default=5.0,
    description="Minimum playtime for a game to be considered",
)
args = parser.parse_args()

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
game_embeddings = np.load(
    os.path.join(
        os.getcwd(),
        "data/training_dataset/game_embeddings_5000_most_played_games_prompts=False.npy",
    )
)
# Drop zero rows
mask = (user_game_matrix == 0.0).all(axis=1)
user_game_matrix = user_game_matrix[~mask]


def main(
    user_game_matrix=user_game_matrix,
    game_information=game_information,
    game_embeddings=game_embeddings,
):
    # Subset into acitve users according to criteria
    active_users = sample_users_having_at_least_n_games(
        args.min_games, args.min_playtime, user_game_matrix
    )
    # Sample n test users
    test_users = active_users.sample(
        args.n_test_users, replace=False, random_state=args.seed
    )
    # Leave the rest as non_test users
    non_test_users = active_users.drop(test_users.index.tolist())
    test_users = test_users.reset_index(drop=True)
    non_test_users = non_test_users.reset_index(drop=True)
    # Load model
    trained_model = torch.load(args.model_path)
    model_inference = CollabNNInference(
        trained_model.user_factors,
        trained_model.game_factors,
        trained_model.idx_to_app_id,
        trained_model.app_id_to_idx,
        trained_model.reference_dataset,
        binary_classification=True,
    )
    model_inference.hidden = trained_model.hidden
    model_inference.out = trained_model.out
    evaluate_recommender(
        recommendation_engine="DeepCollaborativeFiltering",
        test_users=test_users,
        non_test_users=non_test_users,
        occlusion=args.occlusion,
        model=model_inference,
        game_information=game_information,
        game_embeddings=game_embeddings,
        top_k_users=args.n_closest_users,
        seed=args.seed,
        use_content_embeddings=False,
        n_candidates=args.n_candidate_games,
    )
    evaluate_recommender(
        recommendation_engine="DeepCollaborativeFiltering",
        test_users=test_users,
        non_test_users=non_test_users,
        occlusion=args.occlusion,
        model=model_inference,
        game_information=game_information,
        game_embeddings=game_embeddings,
        top_k_users=args.n_closest_users,
        seed=args.seed,
        use_content_embeddings=True,
        n_candidates=args.n_candidate_games,
    )
    evaluate_recommender(
        recommendation_engine="Naive",
        test_users=test_users,
        non_test_users=non_test_users,
        occlusion=args.occlusion,
        top_k_users=args.n_closest_users,
        seed=args.seed,
    )
    evaluate_recommender(
        recommendation_engine="NaiveBinary",
        test_users=test_users,
        non_test_users=non_test_users,
        occlusion=args.occlusion,
        model=model_inference,
        top_k_users=args.n_closest_users,
        seed=args.seed,
    )
    evaluate_recommender(
        recommendation_engine="Content",
        test_users=test_users,
        non_test_users=non_test_users,
        game_information=game_information,
        occlusion=args.occlusion,
        game_embeddings=game_embeddings,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
