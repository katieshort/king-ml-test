"""
This script is the main entry point for training models, making predictions, and evaluating results.

It handles data loading, preprocessing, model training, evaluation, and the generation of recommendations
for game design based on player data. Outputs include model details, propensity scores, and visualizations
of data distributions and recommendations.
"""

import logging
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing.dataloader import DataLoader
from src.models.drlearner import DoubleRobustLearner
from src.models.recommender import GameDesignRecommender
from src.utils.plotting import (
    overlay_design_distributions,
    plot_density_propensity_scores,
    plot_design_distribution,
    plot_distributions_per_group,
    plot_propensity_scores,
)
from src.utils.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_preprocess_data(config: Dict) -> pd.DataFrame:
    """Load and preprocess data."""
    loader = DataLoader(config)
    df = loader.load_data()
    df = loader.preprocess_data(df)
    return df


def train_and_save_model(
    config: Dict, df_train: pd.DataFrame, model_type: str
) -> DoubleRobustLearner:
    """Train a model, evaluate its performance, and save it."""
    logging.info(f"Starting {model_type.capitalize()} model training and evaluation.")
    model = DoubleRobustLearner(config, model_type, verbose=True)
    model.fit(df_train)
    model.evaluate_models()
    logging.info(f"{model_type.capitalize()} model training and evaluation completed.")
    model.save_model()
    return model


def plot_propensity_scores(model: DoubleRobustLearner, save_path: str) -> None:
    """Plot propensity score distributions."""
    propensity_scores_df = model.get_propensity_scores_df()
    plot_density_propensity_scores(
        propensity_scores_df["propensity_score"],
        propensity_scores_df["treatment"],
        save_path,
    )


def make_recommendations(config: Dict, df: pd.DataFrame) -> Any:
    """Load models and make recommendations."""
    recommender = GameDesignRecommender(config)
    recommender.load_model()
    recommendations = recommender.predict(df)
    return recommendations


def plot_results(df: pd.DataFrame):
    """Generate and display plots."""
    plot_design_distribution(df, save_path="reports/figures/design_distribution.png")
    plot_distributions_per_group(df)
    overlay_design_distributions(
        df, save_path="reports/figures/overlay_design_distribution.png"
    )


def main(config_path: str):
    """
    Orchestrates the model training, evaluation, and recommendation processes.

    Loads configuration, preprocesses data, splits the data for training and testing,
    trains models for engagement and monetization, and generates recommendations on the test set.
    Finally, it visualizes the results of the recommendations.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    df = load_and_preprocess_data(config)

    # Split to create final hold-out set not used for cross-fitting the models
    df_train, df_test = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["player_group"]
    )

    # Train models for engagement and monetization
    engagement_model = train_and_save_model(config, df_train, "engagement")
    monetization_model = train_and_save_model(config, df_train, "monetization")

    # Plot propensity scores for both models
    plot_propensity_scores(
        engagement_model, "reports/figures/engagement_propensity_scores.png"
    )
    plot_propensity_scores(
        monetization_model, "reports/figures/monetization_propensity_scores.png"
    )

    # Simulate recommendations on test set
    recommendations = make_recommendations(config, df_test)
    df_test["recommended_design"] = recommendations

    # Plotting results to inspect recommendation distribution
    plot_results(df_test)


if __name__ == "__main__":
    main(config_path="config/config.yaml")
