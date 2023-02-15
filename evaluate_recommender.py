######################################################
#  Evaluate recommenders in a cold start setting     #
######################################################
import argparse
import os

import numpy as np
import pandas as pd
import torch
from models.collaborative_filtering_recommender import CollabNN
from tools.train import evaluate_recommender
from tools.useful_functions import sample_users_having_at_least_n_games

parser = argparse.ArgumentParser(description="Evaluate and train algorithms")
parser.add_argument(
    "-mpath",
    "--model_path",
    type=str,
    default=None,
    help="Path to trained model you want to evaluate",
)
parser.add_argument(
    "-s", "--seed", type=int, default=41, help="Seed for sampling operations"
)
# Occlusion means how many % of the games we try to reconstruct
parser.add_argument(
    "-o",
    "--occlusion",
    type=float,
    default=0.3,
    help="Level of occlusion of the test user",
)
parser.add_argument(
    "-mg",
    "--min_games",
    type=int,
    default=20,
    help="Minimum amount of games a user need to have to be considered",
)
parser.add_argument(
    "-nu", "--n_test_users", type=int, default=20, help="Amount of test users"
)
parser.add_argument(
    "-ncu",
    "--n_closest_users",
    type=int,
    default=13,
    help="For the deep and naive recommender ranking stage, how many users should be used to built the mean embedding",
)
parser.add_argument(
    "-ncand",
    "--n_candidate_games",
    type=int,
    default=120,
    help="For the deep recommender retrival stage - How many candidate games to retrieve",
)
parser.add_argument(
    "-mp",
    "--min_playtime",
    type=float,
    default=5.0,
    help="Minimum playtime for a game to be considered",
)
args = parser.parse_args()

# Load files
game_information = pd.read_parquet(
    os.path.join(
        os.getcwd(),
        "scripts/app/files/data/subset_game_information_5000_most_played_games_prompts=False.parq",
    )
)
user_game_matrix = pd.read_parquet(
    os.path.join(
        os.getcwd(),
        "scripts/app/files/data/subset_user_game_matrix_5000_most_played_games.parq",
    )
)
game_embeddings = np.load(
    os.path.join(
        os.getcwd(),
        "scripts/app/files/data/game_embeddings_5000_most_played_games_prompts=False.npy",
    )
)
game_embeddings_prompts = np.load(
    os.path.join(
        os.getcwd(),
        "data/training_dataset/game_embeddings_5000_most_played_games_prompts=True.npy",
    )
)
# Drop zero rows
mask = (user_game_matrix == 0.0).all(axis=1)
user_game_matrix = user_game_matrix[~mask]


# Load models
# Deep Recommender Binary with Content Embeddings 
model_binary_content = torch.load(os.path.join(os.getcwd(), "scripts/app/files/models/CollabNN/Deep_Recommender_binary_with_content_embeddings_9J2J8.pt"), map_location=torch.device('cpu'))
# Deep Recommender Binary without Content Embeddings 
model_binary_no_content = torch.load(os.path.join(os.getcwd(), "scripts/app/files/models/CollabNN/07_02_2023_13_54-collabnn-I7ULX-bcp=True-n_games=5000-n_users=11727-val_loss=0.3595-best_hit_rate=0.3200-diversity=0.1730-lr=0.0001-momentum=0.0-wd=0.0-top_k_users=5-min_games=20-min_playtimes=5.0-n_negative_samples=4.pt"), map_location=torch.device('cpu'))

def main(
    user_game_matrix=user_game_matrix,
    game_information=game_information,
    game_embeddings=game_embeddings,
    game_embeddings_prompts=game_embeddings_prompts,
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
    evaluate_recommender(
        recommendation_engine="DeepCollaborativeFiltering",
        test_users=test_users,
        non_test_users=non_test_users,
        occlusion=args.occlusion,
        model=model_binary_content,
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
        model=model_binary_no_content,
        game_information=game_information,
        game_embeddings=game_embeddings,
        top_k_users=args.n_closest_users,
        seed=args.seed,
        use_content_embeddings=False,
        n_candidates=args.n_candidate_games,
    )
    # evaluate_recommender(
    #     recommendation_engine="Naive",
    #     test_users=test_users,
    #     non_test_users=non_test_users,
    #     occlusion=args.occlusion,
    #     top_k_users=args.n_closest_users,
    #     seed=args.seed,
    # )
    # evaluate_recommender(
    #     recommendation_engine="NaiveBinary",
    #     test_users=test_users,
    #     non_test_users=non_test_users,
    #     occlusion=args.occlusion,
    #     top_k_users=args.n_closest_users,
    #     seed=args.seed,
    # )
    # evaluate_recommender(
    #     recommendation_engine="Content",
    #     test_users=test_users,
    #     non_test_users=non_test_users,
    #     game_information=game_information,
    #     occlusion=args.occlusion,
    #     game_embeddings=game_embeddings,
    #     seed=args.seed,
    # )
    # evaluate_recommender(
    #     recommendation_engine="Content",
    #     test_users=test_users,
    #     non_test_users=non_test_users,
    #     game_information=game_information,
    #     occlusion=args.occlusion,
    #     game_embeddings=game_embeddings_prompts,
    #     seed=args.seed,
    # )


if __name__ == "__main__":
    main()
