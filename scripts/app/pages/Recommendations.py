################################################################################
# Third page of the App, here we make recommendations based on the user inputs #
################################################################################
import warnings

warnings.filterwarnings(
    "ignore",
    message="`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).",
)
import os

import streamlit as st
from streamlit_helpers.load_data import load_dataframe, load_model, load_numpy
from streamlit_helpers.load_elements import load_elements_to_list
from tools.inference import (
    content_based_recommender,
    make_recommendations_deep_collaborative_filtering,
    naive_recommender_binary_input,
)

base_path = os.getcwd()
# load game containing all the side information
game_informations = load_dataframe(
    path=os.path.join(
        base_path,
        "files/data/subset_game_information_5000_most_played_games_prompts=False.parq",
    )
)
# Load original user game matrix for naive recommender
user_game_matrix = load_dataframe(
    path=os.path.join(
        base_path, "files/data/subset_user_game_matrix_5000_most_played_games.parq"
    )
)
# Load content-based game embeddings for content based recommender
game_embeddings = load_numpy(
    path=os.path.join(
        base_path,
        "files/data/game_embeddings_5000_most_played_games_prompts=False.npy",
    )
)
# Get all genres for the app
genres = load_elements_to_list(game_informations["single_genre"], unique=True)
# Get all game names for the app
games = load_elements_to_list(game_informations["name"], unique=False)
# Call cached data such that they are available during this page
if "model" not in st.session_state:
    model = load_model(
        path=os.path.join(
            base_path,
            "files/models/model_binary_content_RB6A7.pt",
        )
    )
else:
    model = st.session_state["model"]
if "already_have" not in st.session_state:
    already_have = []
else:
    already_have = st.session_state["already_have"]
if "user" not in st.session_state:
    user = []
else:
    user = st.session_state["user"]

# Get appids of selected apps
already_have_ids = game_informations[game_informations["name"].isin(already_have)][
    "appid"
].tolist()
# Make sure appids are strings
already_have_ids = [str(i) for i in already_have_ids]
# Stored set number of recommendations into variable n
n = st.number_input(
    "How many recommendations do you want to get?", value=3, min_value=1, max_value=10
)
# Title for selection process of recommenders
st.markdown(
    "Please select the recommenders to get their recommendations based on your input"
)
# Select box 1
show_content_based = st.checkbox("Content-Based Recommender")
if show_content_based:
    if len(already_have) > 0:
        # Get recommendations based on content based recommender
        recommendations = content_based_recommender(
            already_have_ids=already_have_ids,
            game_information=game_informations,
            game_embeddings=game_embeddings,
            num_recommendations=n,
            return_all=False,
        )
        # Catch situation where there are fewer recommendations than selected due to the fact that users don't have played enough games
        if len(recommendations) < n:
            n = len(recommendations)
        # Subset data
        relevant_features = game_informations.loc[
            game_informations["appid"].isin(recommendations)
        ].copy()
        # Catch bug
        relevant_features = relevant_features.drop_duplicates(
            subset="name", keep="first"
        )
        if n > len(relevant_features):
            n = len(relevant_features)
        # Display images and open store page when user clicks on link
        for i in range(n):
            st.markdown(
                f"[![Recommendation]({relevant_features.iloc[i,:]['header_image'].format(i + 1)})](https://store.steampowered.com/app/{relevant_features.iloc[i,:]['appid']}/)"
            )
    else:
        st.write(
            "Did not receive any owned games. Please go back to the Welcome page and follow the instructions."
        )
# Select box 2
show_collaborative_filtering = st.checkbox("Collaborative Filtering Recommender")
if show_collaborative_filtering:
    if len(already_have) > 0:
        with st.spinner("Loading Recommendations from Collaborative Filtering:"):
            # Get recommendations based on naive recommender
            recommendations = naive_recommender_binary_input(
                already_have_ids,
                user_game_matrix,
                num_recommendations=n,
                top_k_users=5,
            )
            # Get images of recommmendations
            images = game_informations.loc[
                game_informations["appid"].isin(recommendations)
            ]["header_image"].tolist()
            app_ids = game_informations.loc[
                game_informations["appid"].isin(recommendations)
            ]["appid"].tolist()
        for i in range(n):
            st.markdown(
                f"[![Recommendation]({images[i].format(i + 1)})](https://store.steampowered.com/app/{app_ids[i]}/)"
            )
    else:
        st.write(
            "Did not receive any owned games. Please go back to the Welcome page and follow the instructions."
        )
# Select box 3
show_deep = st.checkbox("Deep Recommender")
if show_deep:
    if len(already_have) > 0:
        with st.spinner("Loading Recommendations from Deep Recommender:"):
            # Get recommendations based on deep learning based recommender
            recommendations = make_recommendations_deep_collaborative_filtering(
                model=model,
                content_embeddings=game_embeddings,
                use_content_embeddings=False,
                already_have_ids=already_have_ids,
                user=user,
                reference_dataset=user_game_matrix,
                n_candidates=120,
                top_k_users=13,
                num_recommendations=n,
                binarize=True,
            )
        # Convert from indices to original app ids
        recommendations = [model.idx_to_app_id[i.item()] for i in recommendations]
        images = game_informations.loc[
            game_informations["appid"].isin(recommendations)
        ]["header_image"].tolist()
        app_ids = game_informations.loc[
            game_informations["appid"].isin(recommendations)
        ]["appid"].tolist()
        for i in range(n):
            st.markdown(
                f"[![Recommendation]({images[i].format(i + 1)})](https://store.steampowered.com/app/{app_ids[i]}/)"
            )
    else:
        st.write(
            "Did not receive any owned games. Please go back to the Welcome page and follow the instructions."
        )
