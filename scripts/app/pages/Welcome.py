######################################################################################################
# Second page of the App, here the user can import their steam profile or manually select some games #
######################################################################################################
import warnings
warnings.filterwarnings("ignore", message="`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).")
import streamlit as st
from tools.useful_functions import get_user_games
import os
from streamlit_helpers.load_data import load_dataframe, load_numpy
from streamlit_helpers.load_elements import load_elements_to_list, build_user_vector

base_path = os.getcwd()
# load game containing all the side information
game_informations = load_dataframe(path=os.path.join(
        base_path,
        "files/data/subset_game_information_5000_most_played_games_prompts=False.parq",
    )
)
# Load original user game matrix for naive recommender
user_game_matrix = load_dataframe(path=os.path.join(
        base_path, "files/data/subset_user_game_matrix_5000_most_played_games.parq"
    )
)
# Load content-based game embeddings for content based recommender
game_embeddings = load_numpy(path=os.path.join(
        base_path,
        "files/data/game_embeddings_5000_most_played_games_prompts=False.npy",
    )
)
# Get all genres for the app
genres = load_elements_to_list(game_informations["single_genre"], unique=True)
# Get all game names for the app
games = load_elements_to_list(game_informations["name"], unique=False)
already_have = []
favourite_genres = []
user = []

# Display title
st.title("Tell us which games you already have")
# Display multiple choice box and save result in choice
choice = st.selectbox(
    "Please choose how you want to import your games",
    options=["Select from list", "Import from Steam"],
)
# Scenario user doesn't import steam profile
if choice == "Select from list":
    st.markdown(
        "Please fill this quick survey and you’ll never need to worry again what to play next!"
    )
    # build to columns to be displayed in app and offer user to pick genres and apps
    col1, col2 = st.columns(2)
    with col1:
        favourite_genres = st.multiselect("Pick your favourite genres", genres)
    with col2:
        already_have = st.multiselect(
            "What games do you already have?",
            games,
        )
# Choice to import profile from Steam
elif choice == "Import from Steam":
    steam_id = st.text_input(
        "Please input your steamid:", value="", placeholder="e.g. 76561198090676153"
    )
    # Case user input a steam id
    if steam_id != "":
        already_have, app_ids, playtimes_user = get_user_games(
            user_id=steam_id,
            user_game_matrix=user_game_matrix,
            only_played=True,
            must_be_present_in_dataset=True,
        )
        user = build_user_vector(user_game_matrix, already_have_ids=app_ids, playtimes=playtimes_user)
        if len(already_have) > 0:
            st.success("Steam games imported", icon="✅")

# Add generated user info to cached data
st.session_state["already_have"] = already_have
st.session_state["favourite_genres"] = favourite_genres
st.session_state["user"] = user