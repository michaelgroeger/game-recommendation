#############################################################################
# Fourth page of the App, here we let the user interact with the embeddings #
#############################################################################
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="`st.experimental_singleton` is deprecated. Please use the new command `st.cache_resource` instead, which has the same behavior. More information [in our docs](https://docs.streamlit.io/library/advanced-features/caching).",
)
import pandas as pd
import plotly.express as px
import streamlit as st
from data_processors.Dataset import get_n_highest_played_games
from streamlit_helpers.load_data import load_dataframe, load_numpy
from streamlit_helpers.load_elements import load_elements_to_list

# Set page name
st.set_page_config(
    page_title="Embedding Explorer",
)
base_path = os.getcwd()
# Call cached data such that they are available during this page
# Initialization already_have
if "already_have" not in st.session_state:
    already_have = []
else:
    already_have = st.session_state["already_have"]
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
# Get base path so app works across systems
base_path = os.getcwd()
# Define mapping for genres such that colors are most distinct
colors = {
    "Action": "#FE0202",
    "Adult Content": "#CA6609",
    "Adventure": "#2EFF00",
    "Casual": "#176605",
    "Early Access": "#798D75",
    "Free to Play": "#00FFF2",
    "Indie": "#05645F",
    "Massively Multiplayer": "#0060FF",
    "RPG": "#5464BC",
    "Racing": "#8C1693",
    "Simulation": "#F100FF",
    "Sports": "#FE9AD8",
    "Strategy": "#000000",
    "Unknown": "#A8A8A8",
    "Utilities": "#746464",
}

# Select embeddings to be displayed
selected_recommenders = st.sidebar.selectbox(
    "Select a recommender", ["Content-Based", "Deep Learning Based"]
)
selected_projection = st.sidebar.selectbox(
    "Select a projection method", ["PCA", "t-SNE"]
)
# Subset by the n most played games
select_n_highest = st.number_input(
    "Inspect Embeddings of n most played games",
    min_value=1,
    max_value=len(game_informations),
    value=len(game_informations),
)
# Scenario user picks content based recommender
if selected_recommenders == "Content-Based":
    # load embedding data for PCA
    if selected_projection == "PCA":
        df = pd.read_parquet(
            os.path.join(
                base_path,
                "files/data/pca_content_embeddings_5000_games.parq",
            )
        )
        # Label the axis
        x_label = "First principal axis"
        y_label = "Second principal axis"
        z_label = "Third principal axis"
    # load embedding data for t-SNE
    elif selected_projection == "t-SNE":
        df = pd.read_parquet(
            os.path.join(
                base_path,
                "files/data/tsne_content_embeddings_5000_games.parq",
            )
        )
        # Label the axis
        x_label = "t-SNE dimension 1"
        y_label = "t-SNE dimension 2"
        z_label = "t-SNE dimension 3"

if selected_recommenders == "Deep Learning Based":
    # load embedding data for PCA
    if selected_projection == "PCA":
        df = pd.read_parquet(
            os.path.join(
                base_path,
                "files/data/PCA_model_embeddings_5000_games.parq",
            )
        )
        # Label the axis
        x_label = "First principal axis"
        y_label = "Second principal axis"
        z_label = "Third principal axis"
    # load embedding data for t-SNE
    elif selected_projection == "t-SNE":
        df = pd.read_parquet(
            os.path.join(
                base_path,
                "files/data/tsne_model_embeddings_5000_games.parq",
            )
        )
        # Label the axis
        x_label = "t-SNE dimension 1"
        y_label = "t-SNE dimension 2"
        z_label = "t-SNE dimension 3"


# Subset in the case user wants to inspect only n-most played games
if select_n_highest != len(game_informations):
    n_most_played_games = get_n_highest_played_games(
        game_informations,
        user_game_matrix,
        n_highest=select_n_highest,
        return_data=False,
    )
    df = df.iloc[n_most_played_games.index.tolist(), :]

df = df.reset_index(drop=True)
filter_genres = st.sidebar.multiselect(
    "Select the genres you want to filter by", genres, default=[]
)
if len(filter_genres) > 0:
    df = df[df["genres"].isin(filter_genres)].reset_index(drop=True)
fig = px.scatter_3d(
    df,
    title=f"{selected_recommenders} {selected_projection} Embeddings",
    x="first_axis",
    y="second_axis",
    z="third_axis",
    color="genres",
    hover_name="names",
    color_discrete_map=colors,
)
mark_game = st.sidebar.selectbox(
    "Select a game you would like to annotate",
    df["names"],
)
game = df[df["names"] == mark_game]
# Plot data
# Improve marker size
fig.update_traces(marker_size=2.5)
game = df[df["names"] == mark_game]
fig.update_layout(
    scene=dict(
        xaxis_title=x_label,
        yaxis_title=y_label,
        zaxis_title=z_label,
        annotations=[
            dict(
                showarrow=True,
                x=float(game["first_axis"]),
                y=float(game["second_axis"]),
                z=float(game["third_axis"]),
                text=str(game["names"].values[0]),
                textangle=0,
                ax=0,
                ay=-75,
                font=dict(color="black", size=12),
                arrowcolor="black",
                arrowsize=3,
                arrowwidth=1,
                arrowhead=1,
            )
        ],
    )
)
st.plotly_chart(fig)
