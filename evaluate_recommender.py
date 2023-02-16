######################################################
#  Evaluate recommenders in a cold start setting     #
######################################################
import argparse
import os
import statistics

import numpy as np
import pandas as pd
import torch
from tools.train import evaluate_recommender
from tools.useful_functions import subset_users_having_at_least_n_games

parser = argparse.ArgumentParser(description="Evaluate and train algorithms")
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
    "-nu", "--n_test_users", type=int, default=50, help="Amount of test users"
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
# Deep Recommender Binary with content embeddings
model_binary_content = torch.load(
    os.path.join(os.getcwd(), "scripts/app/files/models/model_binary_content_RB6A7.pt"),
    map_location=torch.device("cpu"),
)
# Deep Recommender Binary without content embeddings
model_binary_no_content = torch.load(
    os.path.join(
        os.getcwd(), "scripts/app/files/models/model_binary_no_content_MQENC.pt"
    ),
    map_location=torch.device("cpu"),
)
# Deep Recommender Logarithmic without content embeddings
model_logarithmic_no_content = torch.load(
    os.path.join(
        os.getcwd(), "scripts/app/files/models/model_logarithmic_no_content_UAW7Y.pt"
    ),
    map_location=torch.device("cpu"),
)
# Deep Recommender Logarithmic with content embeddings
model_logarithmic_content = torch.load(
    os.path.join(
        os.getcwd(), "scripts/app/files/models/model_logarithmic_content_IE0FA.pt"
    ),
    map_location=torch.device("cpu"),
)


def main(
    user_game_matrix=user_game_matrix,
    game_information=game_information,
    game_embeddings=game_embeddings,
    game_embeddings_prompts=game_embeddings_prompts,
    seeds=[41, 0, 34],
):
    results = {
        "Deep Recommender binary with content embeddings": [],
        "Deep Recommender binary w/o content embeddings": [],
        "Deep Recommender log with content embeddings": [],
        "Deep Recommender log w/o content embeddings": [],
        "Naive Collaborative Filtering Recommender no binary": [],
        "Naive Collaborative Filtering Recommender binary": [],
        "Content Based Recommender w/o prompts": [],
        "Content Based Recommender with prompts": [],
    }
    for seed in seeds:
        # Subset into acitve users according to criteria
        active_users = subset_users_having_at_least_n_games(
            args.min_games, args.min_playtime, user_game_matrix
        )
        # Sample n test users
        test_users = active_users.sample(
            args.n_test_users, replace=False, random_state=seed
        )
        # Leave the rest as non_test users
        non_test_users = active_users.drop(test_users.index.tolist())
        test_users = test_users.reset_index(drop=True)
        non_test_users = non_test_users.reset_index(drop=True)
        # Append results into dict evaluate_recommender -> (mean_accuracy, mean_diversity)
        results["Deep Recommender binary with content embeddings"].append(
            evaluate_recommender(
                recommendation_engine="DeepCollaborativeFiltering",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                model=model_binary_content,
                game_information=game_information,
                game_embeddings=game_embeddings,
                top_k_users=args.n_closest_users,
                seed=seed,
                use_content_embeddings=False,
                n_candidates=args.n_candidate_games,
            )
        )
        results["Deep Recommender binary w/o content embeddings"].append(
            evaluate_recommender(
                recommendation_engine="DeepCollaborativeFiltering",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                model=model_binary_no_content,
                game_information=game_information,
                game_embeddings=game_embeddings,
                top_k_users=args.n_closest_users,
                seed=seed,
                use_content_embeddings=False,
                n_candidates=args.n_candidate_games,
            )
        )
        results["Deep Recommender log w/o content embeddings"].append(
            evaluate_recommender(
                recommendation_engine="DeepCollaborativeFiltering",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                model=model_logarithmic_no_content,
                game_information=game_information,
                game_embeddings=game_embeddings,
                top_k_users=args.n_closest_users,
                seed=seed,
                use_content_embeddings=False,
                n_candidates=args.n_candidate_games,
            )
        )
        results["Deep Recommender log with content embeddings"].append(
            evaluate_recommender(
                recommendation_engine="DeepCollaborativeFiltering",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                model=model_logarithmic_content,
                game_information=game_information,
                game_embeddings=game_embeddings,
                top_k_users=args.n_closest_users,
                seed=seed,
                use_content_embeddings=False,
                n_candidates=args.n_candidate_games,
            )
        )
        results["Naive Collaborative Filtering Recommender no binary"].append(
            evaluate_recommender(
                recommendation_engine="Naive",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                top_k_users=args.n_closest_users,
                seed=seed,
            )
        )
        results["Naive Collaborative Filtering Recommender binary"].append(
            evaluate_recommender(
                recommendation_engine="NaiveBinary",
                test_users=test_users,
                non_test_users=non_test_users,
                occlusion=args.occlusion,
                top_k_users=args.n_closest_users,
                seed=seed,
            )
        )
        results["Content Based Recommender w/o prompts"].append(
            evaluate_recommender(
                recommendation_engine="Content",
                test_users=test_users,
                non_test_users=non_test_users,
                game_information=game_information,
                occlusion=args.occlusion,
                game_embeddings=game_embeddings,
                seed=seed,
            )
        )
        results["Content Based Recommender with prompts"].append(
            evaluate_recommender(
                recommendation_engine="Content",
                test_users=test_users,
                non_test_users=non_test_users,
                game_information=game_information,
                occlusion=args.occlusion,
                game_embeddings=game_embeddings_prompts,
                seed=seed,
            )
        )
    # process results and generate statistics
    print(f"###################################################")
    print(f"################## Final Results ##################")
    print(f"###################################################")
    for key, values in results.items():
        # Separate accuracy and diversity into two arrays and report statistics
        accuracy = []
        diversity = []
        sum = []
        for value in values:
            accuracy.append(value[0])
            diversity.append(value[1])
            total = value[0] + value[1]
            sum.append(total)
        print(
            f"Statistics for : {key} \nmean accuracy across all seeds of {statistics.mean(accuracy)} with std of {statistics.stdev(accuracy)} \nmean diversity across all seeds of {statistics.mean(diversity)} with std of {statistics.stdev(diversity)} \nmean total score across all seeds of {statistics.mean(sum)} with std of {statistics.stdev(sum)}\n"
        )


if __name__ == "__main__":
    main()
