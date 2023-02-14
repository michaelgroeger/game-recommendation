import streamlit as st
import pandas as pd
import torch
import numpy as np

@st.cache_data(ttl=120, show_spinner=False)
def load_dataframe(path, parquet=True):
    if parquet == True:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_numpy(path):
    arr = np.load(path)
    return arr

@st.cache_resource(ttl=120, show_spinner=False)
def load_model(path):
    model = torch.load(path, map_location=torch.device("cpu"))
    return model