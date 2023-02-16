#################################################
# Functions and classes to train the algorithms #
#################################################
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from data_processors.Dataset import HitRatioDataset
from tools.evaluation_cold_start import evaluate_recommender
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_hit_rate(dataset, model, top_k):
    """Calculate the hit rate for a model and dataset

    Args:
        dataset (HitRateDataset): Dataset that returns x negatives for one positive which needs to be reranked by the model
        model: Deep learning based collaborative filtering model
        top_k (int): Take top k ranked games to see if positive is in it

    Returns:
        float, float: Returns the hit rate for the given parameters and inputs and the diversity score
    """
    # get user_id and _app_id
    users = dataset.epoch_dataset["user_id"].unique()
    hits = 0
    diversity = []
    for i in tqdm(users, desc="Getting hit rate"):
        # Retrieve batch per user
        user_ids, app_ids, playtimes = dataset[i]
        # Get game which is played (ground truth)
        gt = app_ids[(playtimes > 0).nonzero(as_tuple=True)[0]].item()
        # Rerank inputs
        outputs = model(user_ids, app_ids)
        # Get top scored items
        _, indices = torch.topk(outputs, top_k)
        # get appids from top scored items
        apps = [app_ids[i].item() for i in indices.tolist()]
        # Append to diversity to calculate score later
        diversity.append(apps)
        # Count it if gt in top ranked items
        if gt in apps:
            hits += 1
    # Calculte, report and return metrics
    hit_rate = hits / len(users)
    diversity = [i for sublist in diversity for i in sublist]
    diversity_score = len(set(diversity)) / len(diversity)
    print(f"Hit rate is {hit_rate}")
    print(f"Diversity is {diversity_score}")
    return hit_rate, diversity_score


def get_train_test_val_of_dataframe(df: pd.DataFrame):
    """Train/Test/Val split of dataframe

    Args:
        df (pd.DataFrame): Dataframe containing data

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: train dataset (60 %), test dataset (20 %), val dataset (20 %))
    """
    train, validate, test = np.split(
        df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))]
    )
    train, validate, test = (
        train.reset_index(drop=True),
        validate.reset_index(drop=True),
        test.reset_index(drop=True),
    )
    return train, validate, test


def train_test_validate(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hit_rate_dataset: HitRatioDataset,
    n_epochs: int,
    lr: float,
    test_users: pd.DataFrame,
    game_information: pd.DataFrame,
    weight_decay: float = 0.0,
    device: str = "cpu",
    top_k_users: int = 5,
    model_path: str = None,
    checkpointing: bool = True,
    lr_scheduling: bool = True,
    binary_classification: bool = False,
    use_wandb: bool = False,
    identifier: str = "no_identifier",
):
    """Trains a deep collaborative filtering model based on inputs and parameters

    Args:
        model: Model instance to be trained
        train_loader (DataLoader): Dataloader for respective training stage
        val_loader (DataLoader): Dataloader for respective training stage
        test_loader (DataLoader): Dataloader for respective training stage
        hit_rate_dataset (HitRatioDataset): Dataset to calculate HitRatio
        n_epochs (int): How long to train
        lr (float): Learning rate
        test_users (pd.DataFrame): Test users
        game_information (pd.DataFrame): Game side information
        weight_decay (float, optional): Weight Decay. Defaults to 0.0.
        device (str, optional): Which device to use. Defaults to "cpu".
        top_k_users (int, optional): Closest users to consider. Defaults to 5.
        model_path (str, optional): Where to save best model when using checkpointing. Defaults to None.
        checkpointing (bool, optional): Whether to use checkpointing. Defaults to True.
        lr_scheduling (bool, optional): Whether to use learning rate scheduling. Defaults to True.
        binary_classification (bool, optional): Whether we are performing binary classification. Defaults to False.
        use_wandb (bool, optional): Whether to track experiment in weights and biases. Defaults to False.
        identifier (str, optional): Unique identifier for a model. Defaults to "no_identifier".

    Returns:
        _type_: _description_
    """
    # Define loss function
    if binary_classification == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # to collect hit rates
    hit_rates = []
    all_diversity = []
    # define monitoring stats in a way they are overwritten later
    best_val_loss = 10000000000
    best_hit_rate = 0.0
    best_cold_start_overall = 0.0
    model_name_old = None
    model, criterion = model.to(device), criterion.to(device)
    if model.game_content_embeddings is not None:
        model.game_content_embeddings = model.game_content_embeddings.to(device)
    # Setup Learning rate scheduling
    T_max = len(train_loader) * n_epochs
    if lr_scheduling == True:
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, verbose=False)
    # iGet date for model name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    # Get initial performance on metrics
    hit_rate, diversity_score = get_hit_rate(hit_rate_dataset, model, 10)
    hit_rates.append(hit_rate)
    all_diversity.append(diversity_score)
    # Get initial accuracy and diversity before training
    model = model.cpu()
    if model.game_content_embeddings is not None:
        model.game_content_embeddings = model.game_content_embeddings.cpu()
    cold_start_accuracy, cold_start_diversity = evaluate_recommender(
        recommendation_engine="DeepCollaborativeFiltering",
        test_users=test_users,
        non_test_users=model.reference_dataset,
        occlusion=0.3,
        model=model,
        game_information=game_information,
        game_embeddings=None,
        verbose=True,
    )
    model = model.to(device)
    if model.game_content_embeddings is not None:
        model.game_content_embeddings = model.game_content_embeddings.to(device)
    if use_wandb == True:
        wandb.log(
            {
                "cold_start_accuracy": cold_start_accuracy,
                "cold_start_diversity": cold_start_diversity,
                "cold_start_overall": cold_start_accuracy + cold_start_diversity,
            }
        )
    # Start training
    for epoch in range(n_epochs):
        model = model.train()
        batch, train_mean_loss = 0, 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for user_ids, app_ids, playtime in tepoch:
                user_ids, app_ids, playtime = (
                    user_ids.to(device),
                    app_ids.to(device),
                    playtime.to(device),
                )
                batch += 1
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(user_ids, app_ids)
                # Compute the loss
                condition = len(torch.gt(outputs, 1.0).nonzero()) == 0
                # Clamp if outputs are greater than one
                if condition == False:
                    torch.gt(outputs, 1.0).nonzero()
                    outputs = torch.clamp(outputs, 0.0, 1.0)
                loss = criterion(outputs, playtime)
                # Backward pass and optimization
                loss.backward()
                # Optimization step
                optimizer.step()
                # Learning rate scheduling step
                if lr_scheduling == True:
                    scheduler.step()
                # Add loss to report later
                train_mean_loss += loss
                # print statistics in progress bar
                tepoch.set_postfix(
                    epoch=epoch,
                    phase="Training",
                    loss=loss.item(),
                    epoch_loss=(train_mean_loss / batch).item(),
                )
        # Get mean loss
        train_mean_loss = (train_mean_loss / batch).item()

        model = model.eval()
        # Validation loop
        with torch.no_grad():
            batch, val_mean_loss = 0, 0.0
            with tqdm(val_loader, unit="batch") as tepoch:
                for user_ids, app_ids, playtime in tepoch:
                    user_ids, app_ids, playtime = (
                        user_ids.to(device),
                        app_ids.to(device),
                        playtime.to(device),
                    )
                    batch += 1
                    outputs = model(user_ids, app_ids)
                    condition = len(torch.gt(outputs, 1.0).nonzero()) == 0
                    # Clamp if outputs are greater than one
                    if condition == False:
                        torch.gt(outputs, 1.0).nonzero()
                        outputs = torch.clamp(outputs, 0.0, 1.0)
                    loss = criterion(outputs, playtime)
                    val_mean_loss += loss
                    # print statistics to progress bar
                    tepoch.set_postfix(
                        epoch=epoch,
                        phase="Validation",
                        loss=loss.item(),
                        epoch_loss=(val_mean_loss / batch).item(),
                    )
        val_mean_loss = (val_mean_loss / batch).item()
        hit_rate, diversity_score = get_hit_rate(hit_rate_dataset, model, 10)
        hit_rates.append(hit_rate)
        all_diversity.append(diversity_score)
        if use_wandb == True:
            wandb.log(
                {
                    "train_loss": train_mean_loss,
                    "val_loss": val_mean_loss,
                    "hit_rate": hit_rate,
                    "diversity": diversity_score,
                }
            )

        # Print summary statistics of epoch
        if lr_scheduling == True:
            print(
                f"Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}, Loss: {train_mean_loss:.4f}, Validation Loss: {val_mean_loss:.4f}"
            )
            if use_wandb == True:
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
        else:
            print(
                f"Epoch: {epoch}, Loss: {train_mean_loss:.4f}, Validation Loss: {val_mean_loss:.4f}"
            )

        # Send model to cpu since cold start benchmarking not built for gpu
        model = model.cpu()
        if model.game_content_embeddings is not None:
            model.game_content_embeddings = model.game_content_embeddings.cpu()
        # Get cold start accuracy and diversity
        cold_start_accuracy, cold_start_diversity = evaluate_recommender(
            recommendation_engine="DeepCollaborativeFiltering",
            test_users=test_users,
            non_test_users=model.reference_dataset,
            occlusion=0.3,
            model=model,
            game_information=game_information,
            game_embeddings=None,
            verbose=True,
        )
        cold_start_overall = cold_start_accuracy + cold_start_diversity
        model = model.to(device)
        if model.game_content_embeddings is not None:
            model.game_content_embeddings = model.game_content_embeddings.to(device)
        if use_wandb == True:
            wandb.log(
                {
                    "cold_start_accuracy": cold_start_accuracy,
                    "cold_start_diversity": cold_start_diversity,
                    "cold_start_overall": cold_start_overall,
                }
            )

        # Do checkpointing
        if checkpointing == True:
            # If we got best loss and best hit rate
            if cold_start_overall > best_cold_start_overall:
                if model_name_old != None and Path(model_name_old).is_file():
                    # Remove old best model
                    os.remove(model_name_old)
                # Overwrite
                best_cold_start_overall = cold_start_overall
                # create model name
                model_name = f"{dt_string}-{model.model_name}-{identifier}-binary={binary_classification}-n_games={model.n_games}-n_users={model.n_users}-best_cold_start_overall={best_cold_start_overall}-acc={cold_start_accuracy}-diversity={cold_start_diversity}-lr={lr}-wd={weight_decay}-top_k_users={top_k_users}-min_games={model.min_games}-min_playtimes={model.min_playtime}-n_negative_samples={model.n_negative_samples}.pt"
                print(
                    f"Found best checkpoint with val_loss {best_val_loss:.4f} and custom test acc {best_hit_rate:.4f}, will export model to {os.path.join(model_path, model_name)}"
                )
                # Export
                torch.save(model, os.path.join(model_path, model_name))
                model_name_old = os.path.join(model_path, model_name)
        # Resample the negatives so we show diverse negatives for each user
        train_loader.dataset.add_negatives()
        hit_rate_dataset.add_negatives()

    # Test loop
    with torch.no_grad():
        test_loss = 0.0
        for user_ids, app_ids, playtime in test_loader:
            user_ids, app_ids, playtime = (
                user_ids.to(device),
                app_ids.to(device),
                playtime.to(device),
            )
            batch += 1
            outputs = model(user_ids, app_ids)
            condition = len(torch.gt(outputs, 1.0).nonzero()) == 0
            if condition == False:
                torch.gt(outputs, 1.0).nonzero()
                outputs = torch.clamp(outputs, 0.0, 1.0)
            loss = criterion(outputs, playtime)
            test_loss += criterion(outputs, playtime)
    test_loss = test_loss / len(test_loader)
    if use_wandb == True:
        wandb.log({"test_loss": test_loss})
    print(f"Test Loss: {test_loss:.4f}")
