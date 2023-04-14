#########################################################################
# Functions for app that processes that and put the results into cache  #
#########################################################################
import pandas as pd
import streamlit as st


@st.cache_data(ttl=120, show_spinner=False)
def load_elements_to_list(series: pd.Series, unique: bool = False) -> list:
    if unique == True:
        return series.unique().tolist()
    else:
        return series.tolist()


@st.cache_data(ttl=120, show_spinner=False)
def load_elements_to_list(series: pd.Series, unique: bool = False) -> list:
    if unique == True:

        return series.unique().tolist()
    else:
        return series.tolist()


@st.cache_data(ttl=120, show_spinner=False)
def build_user_vector(user_game_matrix, already_have_ids, playtimes):
    user_body = user_game_matrix.iloc[0, :].copy()
    user_body[user_body > 0] = 0
    # convert playtimes to float
    playtimes = [float(i) for i in playtimes]
    user_body[already_have_ids] = playtimes
    return user_body
