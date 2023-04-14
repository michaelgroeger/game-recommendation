######################################################################################################
# Second page of the App, here the user can import their steam profile or manually select some games #
######################################################################################################
import warnings

warnings.filterwarnings(
    "ignore",
    message="`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).",
)
import os

import streamlit as st
from streamlit_helpers.load_data import load_dataframe, load_numpy
from streamlit_helpers.load_elements import build_user_vector, load_elements_to_list
from tools.useful_functions import get_user_games

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
if "already_have" not in st.session_state:
    already_have = []
else:
    already_have = st.session_state["already_have"]
# Initialization
if "user" not in st.session_state:
    user = []
else:
    user = st.session_state["user"]

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
        "Pick a few games and you'll never need to worry about what to play next!"
    )
    # offer user to pick games
    already_have = st.multiselect(
        "Which games do you play?",
        games,
        help="If you want games from a certain genre consider putting those which you play and are of that genre.",
    )
    if len(already_have) > 0:
        st.write(
            "When you selected all relevant games, please proceed to the Recommendations via the side bar."
        )
# Choice to import profile from Steam
elif choice == "Import from Steam":
    try:
        steam_id = st.text_input(
            "Please input your steamid:", value="", placeholder="e.g. 76561198170260211"
        )
        # Case user input a steam id
        if steam_id != "":
            already_have, app_ids, playtimes_user = get_user_games(
                user_id=steam_id,
                user_game_matrix=user_game_matrix,
                only_played=True,
                must_be_present_in_dataset=True,
            )
            user = build_user_vector(
                user_game_matrix, already_have_ids=app_ids, playtimes=playtimes_user
            )
            if len(already_have) > 0:
                st.success(
                    "Steam games imported, please proceed to the Recommendations via the side bar",
                    icon="✅",
                )
    except:
        st.success("Unfortunately this didn't work.", icon="❌")
        st.markdown(
            "Please make sure you entered a valid Steam ID. This link will tell you how you can find yours: [Link](https://www.ubisoft.com/en-gb/help/article/finding-your-steam-id/000060565#:~:text=To%20view%20your%20Steam%20ID%3A&text=Select%20your%20Steam%20username.&text=Locate%20the%20URL%20field%20beneath,the%20end%20of%20the%20URL.)"
        )
        st.markdown(
            "Please make also sure your games are visible. Here you can find instruction on how to make your games publicly visible: [Link](https://asapguide.com/how-to-make-steam-profile-public/)"
        )
        st.markdown("Follow the steps and feel free to try again.")

# Add generated user info to cached data
st.session_state["already_have"] = already_have
st.session_state["user"] = user
