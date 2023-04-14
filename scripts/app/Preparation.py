#######################################################################################################################################
# First page of the App, here we define the naming of the pages and load the data which will be accessible during the app interaction #
#######################################################################################################################################
import os

import randfacts
import streamlit as st
from st_pages import Page, show_pages
from streamlit_helpers.load_data import load_dataframe, load_model, load_numpy
from streamlit_helpers.load_elements import load_elements_to_list

# Set name of current page
# st.set_page_config(
#     page_title="Preparation",
# )
# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be
show_pages(
    [
        Page("Preparation.py", "Preparation"),
        Page("pages/Welcome.py", "Welcome"),
        Page("pages/Recommendations.py", "Recommendations"),
        Page("pages/Embedding Explorer.py", "Embedding Explorer"),
    ]
)
# Display title in app
st.title(
    "Welcome to the awesome gaming recommendation app, we are setting everything up for your visit"
)
# Text under title
st.markdown(
    "In the meanwhile learn something new with some random facts. Did you know that: "
)
# Instatiate the place to display the facts
facts = st.empty()
## Loading data
# Set base path to make it work across different systems
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
games = load_elements_to_list(game_informations["name"], unique=True)
# Display sucess message
st.success("Database is ready", icon="✅")
# Display a random fact to ease waiting time for user
facts.write(randfacts.get_fact())
# load the model for the deep recommender
model = load_model(
    path=os.path.join(
        base_path,
        "files/models/model_binary_content_RB6A7.pt",
    )
)
# Display sucess message
st.success("Models are ready to help you finding your next game", icon="✅")
st.success("We are ready to go, please proceed to the Welcome page", icon="🎉")
