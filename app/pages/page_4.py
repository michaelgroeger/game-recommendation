#############################################################################
# Fourth page of the App, here we let the user interact with the embeddings #
#############################################################################
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from data_processors.Dataset import get_n_highest_played_games

# Call cached data such that they are available during this page
genres = st.session_state["genres"]
game_informations = st.session_state["game_informations"]
user_game_matrix = st.session_state["user_game_matrix"]
games = st.session_state["games"]

# Set page name
st.set_page_config(
    page_title="Embedding Explorer",
)
# Get base path so app works across systems
base_path = os.getcwd()

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
                "app/files/data/pca_content_embeddings_5000_games.parq",
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
                "app/files/data/tsne_content_embeddings_5000_games.parq",
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
                "app/files/data/PCA_model_embeddings_5000_games.parq",
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
                "app/files/data/tsne_model_embeddings_5000_games.parq",
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
    color_discrete_sequence=df["color"],
)
# allow_markings  = st.sidebar.button("Find game in embedding view")
# if allow_markings == True:
mark_game = st.sidebar.selectbox(
    "Select a game you would like to annotate",
    df["names"],
)
game = df[df["names"] == mark_game]
# Plot data
# Improve marker size
fig.update_traces(marker_size=2.5)
# Overwrite title and axis
# if allow_markings == False:
#     fig.update_layout(
#         scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
#     )
# else:
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
