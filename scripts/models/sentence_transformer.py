##################################################################
# Instantiates document transformer from:                        #
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2 #
##################################################################
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_model(model: str = "sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model)


def get_content_embeddings(path_to_game_information, path_to_embeddings, model):
    """Reads processed descriptions from game informations, turns them into embeddings and saves them.

    Args:
        path_to_game_information (string): From where to load the game informations
        path_to_embeddings (string): Where to save the embeddings
        model (SentenceTransformer): instance of SentenceTransformer model
    """
    data = pd.read_parquet(path_to_game_information)
    descriptions = data["processed_descriptions"].tolist()
    embeddings = model.encode(descriptions)
    np.save(path_to_embeddings, embeddings)
