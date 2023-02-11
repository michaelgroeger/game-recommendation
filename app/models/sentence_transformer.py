##################################################################
# Instantiates document transformer from:                        #
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2 #
##################################################################
from sentence_transformers import SentenceTransformer


def get_model(model: str = "sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model)
