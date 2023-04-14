import numpy as np
import pandas as pd
import torch
from tools.useful_functions import get_closest_vectors

#################################################################
# Functions and classes to make predictions with the algorithms #
#################################################################


def naive_recommender(
    user: pd.Series,
    non_test_users: pd.DataFrame,
    num_recommendations: int,
    top_k_users: int,
    already_have_ids=None,
):
    """
    Naive recommender as baseline. Takes the entire raw user-test matrix and picks the top k users which are most similar to the user.
    Then means their playtimes and reports the num_recommendations games

    Args:
        user (pd.Series): test user
        non_test_users (pd.DataFrame): non test users. User game matrix
        num_recommendations (int): Top k games to return
        top_k_users (int): Top k similar users to turn to
        already_have_ids (list, optional): List of apps the user already has. Defaults to None.

    Returns:
        top_k_games (list): list of appids which are most played and not owned by user
    """
    # Get closest users
    _, max_indices, _ = get_closest_vectors(
        np.expand_dims(user.to_numpy(), axis=0),
        non_test_users.to_numpy(),
        top_k=top_k_users,
    )
    # Select users by max index
    closest_users = non_test_users.iloc[max_indices, :]
    # mean their playtimes
    mean_user = closest_users.mean(axis=0).sort_values(ascending=False)
    # Get only these games which are not in user
    mean_user_without_alread_have_ids = mean_user.drop(already_have_ids)
    # Return top_k games according to mean playtime
    top_k_games = mean_user_without_alread_have_ids.index.tolist()[:num_recommendations]
    return top_k_games


def build_binary_representations(
    reference_dataset: pd.DataFrame, already_have_ids: list
):
    """Builds a binary representation of a user and a reference user-game matrix

    Args:
        reference_dataset (pd.DataFrame): User Game matrix
        already_have_ids (list): List of apps the user already has.

    Returns:
        inference_user (pd.Series), users_binary (pd.DataFrame): Binarized versions of the test user and the reference user-game matrix
    """
    # Build binary user vector by already_have_ids
    # pick some user as body
    inference_user = reference_dataset.iloc[0, :]
    # Binarize inference user
    inference_user[inference_user > 0.0] = 0.0
    inference_user[inference_user.index.isin(already_have_ids)] = 1.0
    # binarize user-game matrix
    users_binary = reference_dataset.copy()
    users_binary[users_binary > 0.0] = 1.0
    return inference_user, users_binary


def naive_recommender_binary_input(
    already_have_ids: list,
    non_test_users: pd.DataFrame,
    num_recommendations: int,
    top_k_users: int,
    use_binary_times: bool = True,
):
    """
    Naive recommender as baseline but using binary representations. Takes the entire binarized user-test matrix and picks the top k users which are most similar to the binarized user.
    Then means their playtimes and reports the top_k games

    Args:
        already_have_ids (list): List of apps the user already has.
        non_test_users (pd.DataFrame): non test users. User game matrix
        num_recommendations (int): Top k games to return
        top_k_users (int): Top k similar users to select to generate recommendations
        use_binary_times (bool): Whether to use binary times when summarizing the user. If False will use original playtimes.

    Returns:
        mean_user_without_alread_have_ids (list): list of appids which are most played and not owned by user
    """
    # Get binary representations
    inference_user, users_binary = build_binary_representations(
        non_test_users, already_have_ids
    )
    # Get closest users
    _, max_indices, _ = get_closest_vectors(
        np.expand_dims(inference_user.to_numpy(), axis=0),
        users_binary.to_numpy(),
        top_k=top_k_users,
    )
    # Select users by max index, but from original matrix so we have the playtimes
    closest_users = non_test_users.iloc[max_indices, :].copy()
    # Binarize playtimes
    if use_binary_times == True:
        closest_users[closest_users > 0] = 1
    mean_user = closest_users.mean(axis=0).sort_values(ascending=False)
    mean_user_without_alread_have_ids = mean_user.drop(already_have_ids).index.tolist()[
        :num_recommendations
    ]
    # Get suggestions by reporting the sorted indices which correspond with the index in the game information
    return mean_user_without_alread_have_ids


def content_based_recommender(
    already_have_ids,
    game_information,
    game_embeddings,
    num_recommendations=10,
    return_all=False,
):
    """Make recommendations based on a content-based approach.

    Args:
        already_have_ids (list): List of apps the user already has.
        game_information (pd.DataFrame): Table containing game informations
        game_embeddings (np.array): Content based game embeddings
        num_recommendations (int, optional): Top k games to return. Defaults to 10.

    Returns:
        recommendations (list): list of appids which are most played and not owned by user
    """
    # Get owned games
    owned_games = game_information[
        game_information["appid"].isin(already_have_ids)
    ].index.tolist()
    # Build mean vector based on games that the user already has
    mean_owned_games = np.mean(game_embeddings[owned_games, :], axis=0)
    # drop the owned games from the embeddings and the game information table
    game_embeddings_not_owned = np.delete(game_embeddings, owned_games, 0)
    game_information_not_owned = game_information.drop(owned_games).reset_index(
        drop=True
    )
    assert len(game_embeddings_not_owned) == len(
        game_information_not_owned
    ), f"Got {len(game_embeddings_not_owned)} vs. {len(game_information_not_owned)}"
    # Get top_k
    if return_all == True:
        # Case to return all games and then filter in the app
        num_recommendations = len(game_embeddings)
    _, max_indices, _ = get_closest_vectors(
        np.expand_dims(mean_owned_games, axis=0),
        game_embeddings,
        top_k=num_recommendations + len(owned_games),
    )
    max_indices = np.squeeze(max_indices[max_indices != owned_games])
    recommendations = game_information.iloc[max_indices]["appid"].tolist()[
        :num_recommendations
    ]
    return recommendations


def mask_user(user, occlusion, seed=41):
    """
    Masks a user in a sense that it takes out occlusion % of the nonzero entries

    Args:
        user (pd.Series): unseen user
        occlusion (float): how much of the user should be taken out
        seed (int, optional): Seed for sampling operations. Defaults to 41.

    Returns:
        already_have_ids (list), to_be_reconstructed_games (list): List of ids which will be shown to the algorithms. List of ids which should be reconstructed.
    """
    # Split into known games and occluded games
    # sample n games so we delete occulsion % of them wich is then to be reconstruted
    sample_n_games = int(len(user[(user != 0)]) * occlusion)
    take_out_n_games = user[(user != 0)].sample(
        n=sample_n_games, replace=False, random_state=seed
    )
    # Get their ids
    already_have_ids = user[(user != 0)][
        ~user[(user != 0)].isin(take_out_n_games)
    ].index.tolist()
    to_be_reconstructed_games = take_out_n_games.index.tolist()
    return already_have_ids, to_be_reconstructed_games


def retrieval(
    known_games, model, content_embeddings, n_candidates, use_content_embeddings=False
):
    """Retrieve candidate items using either the content based embeddings or the learnt ones from the collaborative filtering recommender.

    Args:
        known_games (list): list of ids in embedding tables
        model: Deep recommender model
        content_embeddings (np.array): Array holding content based embeddings
        n_candidates (int): Number of candidate items to retrieve
        use_content_embeddings (bool, optional): Whether to use the content based embeddings to retrieve candidates. Defaults to False.

    Returns:
        candidate_games (list): list of candidate games to be ranked by model
    """
    if use_content_embeddings == True:
        game_embeddings_already_have_ids = content_embeddings[known_games]
        mean_game_embeddings_already_have_ids = np.mean(
            game_embeddings_already_have_ids, axis=0
        )
        all_embeddings = content_embeddings
    else:
        game_embeddings_already_have_ids = model.game_factors[known_games]
        mean_game_embeddings_already_have_ids = (
            torch.mean(game_embeddings_already_have_ids, dim=0).cpu().detach().numpy()
        )
        game_embeddings_already_have_ids = (
            game_embeddings_already_have_ids.cpu().detach().numpy()
        )
        all_embeddings = model.game_factors.cpu().detach().numpy()

    _, candidate_games, _ = get_closest_vectors(
        np.expand_dims(mean_game_embeddings_already_have_ids, axis=0),
        all_embeddings,
        top_k=n_candidates,
    )
    candidate_games = [i for i in candidate_games if i not in known_games]
    return candidate_games


def rank_candidates(
    user: pd.Series,
    model,
    top_k_users: int,
    reference_dataset: pd.DataFrame,
    candidate_games: list,
    num_recommendations: int,
):
    """Use the deep leanring model to rank items among some candidates

    Args:
        user (pd.Series): unseen user
        model: Deep learning trained model
        top_k_users (int): Top k similar users to turn to
        reference_dataset (pd.DataFrame): User Game matrix
        candidate_games (list): Games returned by retrieval
        num_recommendations (int, optional): Top k games to return. Defaults to 10.

    Returns:
        recommendations (list): list of appids which are most played and not owned by user
    """
    # Retrieve closest users
    _, closest_user, _ = get_closest_vectors(
        np.expand_dims(user.to_numpy(), axis=0),
        reference_dataset.to_numpy(),
        top_k=top_k_users,
    )
    # turn candidate game ids to tensor
    app_ids = torch.LongTensor(candidate_games).squeeze()
    # Make predictions using the model
    predictions = model.forward_cold_start(user_ids=closest_user, app_ids=app_ids)
    # Get top k games and return as recommendation
    _, indices = torch.topk(predictions, k=num_recommendations)
    recommendations = app_ids[indices]
    return recommendations


def make_recommendations_deep_collaborative_filtering(
    model,
    content_embeddings: np.array,
    use_content_embeddings: bool,
    already_have_ids: list,
    user: pd.Series,
    reference_dataset: pd.DataFrame,
    n_candidates: int,
    top_k_users: int,
    num_recommendations: int,
    binarize: bool = False,
):
    """
    Use retrieval and ranking stages to make predicitons for an unseen user.

    Args:
        model: Deep learning based model
        content_embeddings (np.array): Content based game embeddings
        use_content_embeddings (bool): Whether to use the content based embeddings in the retrieval stage
        already_have_ids (list): List of games the user already has
        user (pd.Series): unseen user
        reference_dataset (pd.DataFrame): user-game matrix the model was trained on
        n_candidates (int): How many candidate games to retrieve
        top_k_users (int): How many top closest user to consider to form embedding for unseen user
        num_recommendations (int): Number of recommendations to return
        binarize (bool, optional): Whether to binarize the user and the reference dataset for retrieving the closest users. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Convert the app ids to indices in the embedding tables
    known_games = [model.app_id_to_idx[i] for i in already_have_ids]
    # Generate candidates
    candidate_games = retrieval(
        known_games=known_games,
        model=model,
        content_embeddings=content_embeddings,
        n_candidates=n_candidates,
        use_content_embeddings=use_content_embeddings,
    )
    # Generate recommendations
    if binarize == True:
        user, reference_dataset = build_binary_representations(
            reference_dataset=reference_dataset, already_have_ids=already_have_ids
        )
    recommendations = rank_candidates(
        user=user,
        model=model,
        top_k_users=top_k_users,
        reference_dataset=reference_dataset,
        candidate_games=candidate_games,
        num_recommendations=num_recommendations,
    )
    return recommendations
