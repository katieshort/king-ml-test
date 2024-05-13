import logging
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing.dataloader import DataLoader
from src.models.drlearner import DoubleRobustLearner
from src.models.recommender import GameDesignRecommender
from src.utils.utils import load_config
from src.utils.plotting import (
    plot_design_distribution,
    overlay_design_distributions,
    plot_distributions_per_group,
)

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

def train_and_save_model(config: Dict, df_train: pd.DataFrame, model_type: str) -> DoubleRobustLearner:
    """Train a model, evaluate its performance, and save it."""
    model = DoubleRobustLearner(config, model_type)
    model.fit(df_train)
    model_details = model.evaluate_models()
    logging.info(f"{model_type.capitalize()} Model Details: {model_details}")
    model.save_model()
    return model


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
    config = load_config(config_path)
    df = load_and_preprocess_data(config)
    # Split to create final hold-out set not used for cross-fitting the models
    df_train, df_test = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["player_group"]
    )
    # Train models for engagement and monetization
    model_engagement = train_and_save_model(config, df_train, "engagement")
    model_monetization = train_and_save_model(config, df_train, "monetization")

    # Simulate recommendations on test set
    recommendations = make_recommendations(config, df_test)
    df_test["recommended_design"] = recommendations

    # Plotting results to inspect recommendation distribution
    plot_results(df_test)

if __name__ == "__main__":
    main(config_path="config/config.yaml")
