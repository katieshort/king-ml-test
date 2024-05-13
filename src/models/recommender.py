"""
This module contains the GameDesignRecommender class, which utilizes pre-trained machine
learning models to make game design recommendations based on predicted engagement and monetization outcomes under
design A vs design B.

The recommender takes in player data, applies preprocessing, and uses the trained models
to recommend design changes aimed at optimizing game performance metrics at individual player level.
"""

import logging
from typing import Any, Dict, List
import joblib
import pandas as pd
from src.data_processing.dataloader import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)


class GameDesignRecommender:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GameDesignRecommender with configuration.
        Args:
            config (dict): Configuration dictionary including model paths and feature settings.
        """
        self.config = config
        self.model_engagement = None
        self.model_monetization = None
        self.preprocessor = DataLoader(config)
        self.load_model()

    def load_model(self) -> None:
        """
        Load models from specified file paths in the configuration.
        """
        try:
            self.model_engagement = joblib.load(
                self.config["file_paths"]["model_path"]["engagement"]
            )
            self.model_monetization = joblib.load(
                self.config["file_paths"]["model_path"]["monetization"]
            )
        except Exception as e:
            logging.error(f"Failed to load models: {e}")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the input dataframe.
        """
        required_columns = set(
            self.config["features"]["engagement"]["X"]
            + self.config["features"]["monetization"]["X"]
        )
        if not required_columns.issubset(df.columns):
            logging.warning("Dataframe missing required features")
            return False
        return True

    def predict(self, df: pd.DataFrame) -> List[str]:
        """
        Predict recommendations based on the game design data provided.
        Default to 'A' if any issues occur.
        """
        try:
            df = self.preprocessor.preprocess_data(df, train=False)

            # Validate the processed data
            if not self.validate_data(df):
                logging.error(
                    "Processed data failed validation. Defaulting to 'Design A'."
                )
                return ["A"] * len(df)  # Default to 'A' if validation fails

            X_engagement = df[self.config["features"]["engagement"]["X"]]
            X_monetization = df[self.config["features"]["monetization"]["X"]]

            # Predict the effect of design B over A
            effects_B_over_A_engagement = self.model_engagement.predict(X_engagement)
            effects_B_over_A_monetization = self.model_monetization.predict(
                X_monetization
            )

            # Recommendation based on predicted effects
            # This could be improved by considering the confidence interval, effect size rather than simple rule
            recommendations = [
                "B" if effect_e >= 0 and effect_m > 0 else "A"
                for effect_e, effect_m in zip(
                    effects_B_over_A_engagement, effects_B_over_A_monetization
                )
            ]
            return recommendations

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return ["A"] * len(df)  # Default to A if an exception occurs
