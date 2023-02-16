####################################################
#  Evaluation protocol for the cold start problem  #
####################################################

import numpy as np
import pandas as pd
from tools.inference import (
    content_based_recommender,
    make_recommendations_deep_collaborative_filtering,
    naive_recommender,
    naive_recommender_binary_input,
)
from tqdm import tqdm


def evaluate_recommender(
    recommendation_engine: str,
    test_users: pd.DataFrame,
    non_test_users: pd.DataFrame,
    occlusion: float,
    model=None,
    game_information: pd.DataFrame = None,
    game_embeddings: np.array = None,
    top_k_users=13,
    seed=41,
    use_content_embeddings=False,
    n_candidates=120,
    binarize=False,
    verbose=False,
):
    """Function to evaluate the different recommendation systems simulating a cold start problem

    Args:
        recommendation_engine (str): Name of the model to benchmark
        test_users (pd.DataFrame): Dataframe containing a user-game matrix of test users.
        non_test_users (pd.DataFrame): Dataframe containing a user-game matrix of users that have been used to train a deep learning based model.
        occlusion (float): How much of an user to occlude before feeding the user into the pipeline.
        model: Deep Collaborative Filtering model
        game_information (pd.DataFrame): Dataframe holding game side-features
        game_embeddings (np.array): Array holding the game content embeddings
        top_k_users (int, optional): When looking up for similar users. How many of the closest users should be considered for inference? Defaults to 13.
        seed (int, optional): Seed for sampling operations. Defaults to 41.
        use_content_embeddings (bool, optional): Whether to user content embeddings for retrieval in the collaborative filtering model. Defaults to False.
        n_candidates (int, optional): How many games should be retrieved for the model to be ranked. Defaults to 120.
        binarize (bool, optional): Whether to binarize the test and non-test users to determine their similarity. Defaults to False.
        verbose (bool, optional): Whether to print the results or not. Defaults to False.

    Returns:
        mean_accuracy(float), mean_diversity(float)
    """
    # Collect all accurcay metrics during evaluation
    total_accuracy = 0.0
    # Collect all recommendations
    all_recommendations_for_diversity = []
    for i in tqdm(range(len(test_users)), desc=f"Benchmarking {recommendation_engine}"):
        # Select one test user
        test_user = test_users.iloc[i, :].copy()
        # get all games that were played
        ground_truth_games = test_user[(test_user != 0)]
        # sample n games so we delete occulsion % of them wich is then to be reconstruted
        sample_n_games = int(len(ground_truth_games) * occlusion)
        take_out_n_games = ground_truth_games.sample(
            n=sample_n_games, replace=False, random_state=seed
        )
        # We'll filter the known games out later so they don't occur in the predicions
        # this can lead to an index out of bounds error downstream so we add a reasonable buffer
        if len(ground_truth_games) > n_candidates:
            n_candidates = len(ground_truth_games)
        # Get appids of games that are played
        already_have_ids = ground_truth_games[
            ~ground_truth_games.isin(take_out_n_games)
        ].index.tolist()
        # Get app ids of games that have been occluded
        take_out_n_games = take_out_n_games.index.tolist()
        # We will delete occlusion% of the owned games and reconstruct them later
        test_user[take_out_n_games] = 0.0
        if recommendation_engine == "Naive":
            recommendations = naive_recommender(
                test_user,
                non_test_users,
                num_recommendations=sample_n_games,
                top_k_users=top_k_users,
                already_have_ids=already_have_ids,
            )
        if recommendation_engine == "NaiveBinary":
            recommendations = naive_recommender_binary_input(
                already_have_ids,
                non_test_users,
                num_recommendations=sample_n_games,
                top_k_users=top_k_users,
            )
        elif recommendation_engine == "Content":
            recommendations = content_based_recommender(
                already_have_ids,
                game_information=game_information,
                game_embeddings=game_embeddings,
                num_recommendations=sample_n_games,
            )
        elif recommendation_engine == "DeepCollaborativeFiltering":
            recommendations = make_recommendations_deep_collaborative_filtering(
                model,
                content_embeddings=game_embeddings,
                use_content_embeddings=use_content_embeddings,
                already_have_ids=already_have_ids,
                user=test_users.iloc[i, :],
                reference_dataset=non_test_users,
                n_candidates=n_candidates,
                top_k_users=top_k_users,
                num_recommendations=len(take_out_n_games),
                binarize=binarize,
            )
            # Convert from indices to original app ids
            recommendations = [model.idx_to_app_id[i.item()] for i in recommendations]
        # Get accuracy of reconstruction
        accuracy = len(set(recommendations) & set(take_out_n_games)) / len(
            take_out_n_games
        )
        # Append recommendations for final diversity score
        all_recommendations_for_diversity.append(recommendations)
        # add accuracy to total accuracy
        total_accuracy += accuracy
    # flatten
    all_recommendations_for_diversity = [
        i for sublist in all_recommendations_for_diversity for i in sublist
    ]
    # Get mean accuracy
    mean_accuracy = total_accuracy / len(test_users)
    # Get diversity score
    mean_diversity = len(set(all_recommendations_for_diversity)) / len(
        all_recommendations_for_diversity
    )
    if verbose == True:
        print(f"Mean accurcay is {mean_accuracy}, mean diversity is {mean_diversity}")
    return mean_accuracy, mean_diversity
