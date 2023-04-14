####################################################################################################################################################
# Definitions of the collaborative filtering models which are adapted from these papers:                                                           #
# - DotProductBias: Probabilistc Matrix Factorization (https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)  #
# - CollabNN: Neural Collaborative Filtering (https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)                                                  #
####################################################################################################################################################

import pandas as pd
import torch
import torch.nn as nn


def create_params(size: tuple) -> torch.Tensor:
    """Creates a Tensor of size to be optimized during Training. Initialized
    by drawing from Normal distribution with zero mean and std: 0.01

    Args:
        size (tuple): Size of Tensor

    Returns:
        torch.Tensor: Initialized pytorch tensor
    """
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


class DotProductBias(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_games: int,
        embedding_dim: int,
        idx_to_app_id: dict = None,
        app_id_to_idx: dict = None,
        reference_dataset: pd.DataFrame = None,
        binary_classification: bool = False,
    ):
        """Initalizes a probabilistic matrix factorization model

        Args:
            n_users (int): Length of user factors tensor.
            n_games (int): Length of game factors tensor.
            embedding_dim (int): Num columns of factors tensor.
            idx_to_app_id (dict, optional): Mapping from id to appid. Defaults to None.
            app_id_to_idx (dict, optional): Mapping from appid to id. Defaults to None.
            reference_dataset (pd.DataFrame, optional): Dataset containing the training users. Useful to extract n_closest users to cold-start user in later stages. Defaults to None.
            binary_classification (bool, optional): Whether to perform binary classification. Defaults to False.
        """
        super(DotProductBias, self).__init__()
        self.n_users = n_users
        self.n_games = n_games
        self.embedding_dim = embedding_dim
        self.user_factors = create_params([self.n_users, self.embedding_dim])
        self.game_factors = create_params([self.n_games, self.embedding_dim])
        self.user_bias = create_params([self.n_users])
        self.game_bias = create_params([self.n_games])
        if binary_classification == False:
            self.out = nn.ReLU()
        elif binary_classification == True:
            self.out = nn.Sigmoid()
        # CONVERT MODEL TO DOUBLE TO MATCH DATA TYPE
        self.double()
        # Add mappings so it will be saved alongside the model
        self.idx_to_app_id = idx_to_app_id
        self.app_id_to_idx = app_id_to_idx
        self.model_name = "matrixfacorization"
        self.reference_dataset = reference_dataset

    def forward(self, user_ids, app_ids):
        # Get embeddings
        user_embeddings = self.user_factors[user_ids]
        game_embeddings = self.game_factors[app_ids]
        # Do dot product
        res = (user_embeddings * game_embeddings).sum(dim=1)
        # Add bias
        res += self.user_bias[user_ids] + self.game_bias[app_ids]
        return self.out(res)

    def forward_cold_start(self, user_ids, app_ids):
        """Adapted forward function to perfrom cold-start prediction

        Args:
            user_ids (torch.LongTensor): Indices of user embeddings
            app_ids (torch.LongTensor): Indices of game embeddings

        Returns:
            Logit: Logit of inputs
        """
        # Build a new user embedding based on mean of selected user embeddings
        mean_user_embedding = torch.mean(self.user_factors[user_ids], dim=0)
        # populate mean user embedding such that we can feed it along the n to be predicted apps
        populated_mean_embedding = torch.LongTensor()
        for i in range(len(app_ids)):
            populated_mean_embedding = torch.cat(
                [populated_mean_embedding, mean_user_embedding.unsqueeze(0)], dim=0
            )
        # Select game embeddings
        game_embeddings = self.game_factors[app_ids]
        # Do dot product of mean user embeddings and game embeddings
        res = (populated_mean_embedding * game_embeddings).sum(dim=1)
        # feed into output layer
        output = self.out(res)
        return output.squeeze()


class CollabNN(nn.Module):
    def __init__(
        self,
        n_users,
        n_games,
        embedding_dim,
        game_content_embeddings=None,
        idx_to_app_id=None,
        app_id_to_idx=None,
        reference_dataset=None,
        binary_classification=False,
    ):
        """Initalizes a neural collaborative filtering model

        Args:
            n_users (int): Length of user factors tensor.
            n_games (int): Length of game factors tensor.
            embedding_dim (int): Num columns of factors tensor.
            game_content_embeddings (torch.Tensor): Content embeddings of detailed description of games.
            idx_to_app_id (dict, optional): Mapping from id to appid. Defaults to None.
            app_id_to_idx (dict, optional): Mapping from appid to id. Defaults to None.
            reference_dataset (pd.DataFrame, optional): Dataset containing the training users. Useful to extract n_closest users to cold-start user in later stages. Defaults to None.
            binary_classification (bool, optional): Whether to perform binary classification. Defaults to False.
        """
        super(CollabNN, self).__init__()
        self.n_users = n_users
        self.n_games = n_games
        self.embedding_dim = embedding_dim
        self.user_factors = create_params([self.n_users, self.embedding_dim])
        self.game_factors = create_params([self.n_games, self.embedding_dim])
        self.game_content_embeddings = game_content_embeddings
        if self.game_content_embeddings == None:
            input_features = embedding_dim + embedding_dim
        else:
            input_features = (
                embedding_dim + embedding_dim + self.game_content_embeddings.shape[1]
            )
        # Build hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        )
        if binary_classification == False:
            self.out = nn.ReLU()
        elif binary_classification == True:
            self.out = nn.Sigmoid()
        # CONVERT MODEL TO DOUBLE TO MATCH DATA TYPE
        self.double()
        # Add mappings so it will be saved alongside the model
        self.idx_to_app_id = idx_to_app_id
        self.app_id_to_idx = app_id_to_idx
        self.model_name = "collabnn"
        self.reference_dataset = reference_dataset

    def forward(self, user_ids, app_ids):
        user_embeddings = self.user_factors[user_ids]
        game_embeddings = self.game_factors[app_ids]
        # feed embeddings into hidden layers
        if self.game_content_embeddings is not None:
            game_content_embeddings = self.game_content_embeddings[app_ids]
            output = self.hidden(
                torch.cat(
                    [user_embeddings, game_embeddings, game_content_embeddings], dim=1
                )
            )
        else:
            output = self.hidden(torch.cat([user_embeddings, game_embeddings], dim=1))
        output = self.out(output)
        return output.squeeze()

    def forward_cold_start(self, user_ids, app_ids):
        """Adapted forward function to perfrom cold-start prediction

        Args:
            user_ids (torch.LongTensor): Indices of user embeddings
            app_ids (torch.LongTensor): Indices of game embeddings

        Returns:
            Logit: Logit of inputs
        """
        # Build a new user embedding based on mean of selected user embeddings
        mean_user_embedding = torch.mean(self.user_factors[user_ids], dim=0)
        # populate mean user embedding such that we can feed it along the n to be predicted apps
        populated_mean_embedding = torch.LongTensor()
        for i in range(len(app_ids)):
            populated_mean_embedding = torch.cat(
                [populated_mean_embedding, mean_user_embedding.unsqueeze(0)], dim=0
            )
        # Select game embeddings
        game_embeddings = self.game_factors[app_ids]
        # feed embeddings into hidden layers
        if self.game_content_embeddings is not None:
            game_content_embeddings = self.game_content_embeddings[app_ids]
            output = self.hidden(
                torch.cat(
                    [
                        populated_mean_embedding,
                        game_embeddings,
                        game_content_embeddings,
                    ],
                    dim=1,
                )
            )
        else:
            output = self.hidden(
                torch.cat(
                    [
                        populated_mean_embedding,
                        game_embeddings,
                    ],
                    dim=1,
                )
            )
        # Feed into output layer
        output = self.out(output)
        return output.squeeze()
