######################################################################################################
# Second page of the App, here the user can import their steam profile or manually select some games #
######################################################################################################
import streamlit as st
from tools.useful_functions import get_user_games

# Set page title
st.set_page_config(
    page_title="Welcome",
)

# Instantiate lists to collect user data
already_have = []
favourite_genres = []
# Call cached data such that they are available during this page
genres = st.session_state["genres"]
games = st.session_state["games"]
user_game_matrix = st.session_state["user_game_matrix"]

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
        if len(already_have) > 0:
            st.success("Steam games imported", icon="✅")

# Add generated user info to cached data
st.session_state["already_have"] = already_have
st.session_state["favourite_genres"] = favourite_genres
