######################################################################
# Functions for app that load and cache data to make the app faster  #
######################################################################
import numpy as np
import pandas as pd
import streamlit as st
import torch


## ttl-> After these seconds the file in cache will be renewed with the next interaction
@st.cache_data(ttl=240, show_spinner=False)
def load_dataframe(path, parquet=True):
    if parquet == True:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


@st.cache_data(ttl=240, show_spinner=False)
def load_numpy(path):
    arr = np.load(path)
    return arr


@st.cache_resource(ttl=240, show_spinner=False)
def load_model(path):
    model = torch.load(path, map_location=torch.device("cpu"))
    return model
