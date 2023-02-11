################################################################################
# Third page of the App, here we make recommendations based on the user inputs #
################################################################################
import streamlit as st
from tools.inference import (
    content_based_recommender,
    make_recommendations_deep_collaborative_filtering,
    naive_recommender_binary_input,
)

# Set name of page
st.set_page_config(
    page_title="Recommendations",
)
# Call cached data such that they are available during this page
already_have = st.session_state["already_have"]
favourite_genres = st.session_state["favourite_genres"]
game_informations = st.session_state["game_informations"]
game_embeddings = st.session_state["game_embeddings"]
user_game_matrix = st.session_state["user_game_matrix"]
model = st.session_state["model"]

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
# Load all available links to images such that we can load images of the recommended ones
images = game_informations["header_image"].tolist()
# Select box 1
show_content_based = st.checkbox("Content-Based Recommender")
if show_content_based:
    # Get recommendations based on content based recommender
    recommendations = content_based_recommender(
        already_have_ids=already_have_ids,
        game_information=game_informations,
        game_embeddings=game_embeddings,
        num_recommendations=n,
    )
    # Get images of recommmendations
    images = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "header_image"
    ].tolist()
    # Get original app id to be 100 % sure it works in the link to the store page
    app_ids = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "appid"
    ].tolist()
    # Clean dublicates to catch bug that shows the same id twice
    app_ids = list(set(app_ids))
    # Display images and open store page when user clicks on link
    for i in range(n):
        st.markdown(
            f"[![Recommendation]({images[i].format(i + 1)})](https://store.steampowered.com/app/{app_ids[i]}/)"
        )
# Select box 2
show_collaborative_filtering = st.checkbox("Collaborative Filtering Recommender")
if show_collaborative_filtering:
    # Get recommendations based on naive recommender
    recommendations = naive_recommender_binary_input(
        already_have_ids,
        user_game_matrix,
        num_recommendations=n,
        top_k_users=5,
    )
    # Get images of recommmendations
    images = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "header_image"
    ].tolist()
    app_ids = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "appid"
    ].tolist()
    for i in range(n):
        st.markdown(
            f"[![Recommendation]({images[i].format(i + 1)})](https://store.steampowered.com/app/{app_ids[i]}/)"
        )
# Select box 3
show_deep = st.checkbox("Deep Recommender")
if show_deep:
    # Get recommendations based on deep learning based recommender
    recommendations = make_recommendations_deep_collaborative_filtering(
        model=model,
        content_embeddings=game_embeddings,
        use_content_embeddings=False,
        already_have_ids=already_have_ids,
        user=None,
        reference_dataset=user_game_matrix,
        n_candidates=120,
        top_k_users=13,
        num_recommendations=n,
        binarize=True,
    )
    # Convert from indices to original app ids
    recommendations = [model.idx_to_app_id[i.item()] for i in recommendations]
    images = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "header_image"
    ].tolist()
    app_ids = game_informations.loc[game_informations["appid"].isin(recommendations)][
        "appid"
    ].tolist()
    for i in range(n):
        st.markdown(
            f"[![Recommendation]({images[i].format(i + 1)})](https://store.steampowered.com/app/{app_ids[i]}/)"
        )
