import pandas as pd
import numpy as np
from typing import Tuple, List, Any
import logging


class DataLoader:
    def __init__(self, config: dict, verbose: bool = False, train: bool = True):
        self.config = config
        self.engagement_features = self.config.get("preprocessing", {}).get(
            "engagement_features", {"n1": 30, "n13": 7}
        )
        logging.basicConfig(level=logging.INFO)

    def load_data(self) -> pd.DataFrame:
        """Load data from a specified path in config."""
        try:
            df = pd.read_csv(self.config.get("file_paths", {}).get("data_path"))
            logging.info(
                "Data loaded successfully from {}".format(
                    self.config["file_paths"]["data_path"]
                )
            )
            return df
        except Exception as e:
            logging.error("Failed to load data: {}".format(e))
            raise

    def preprocess_data(self, df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
        """Handle all preprocessing that does not require fitting."""
        df = df.copy()

        if train:
            df = (
                self.filter_negative_engagement(df)
                if self.config["preprocessing"].get("filter_negative_engagement", False)
                else df
            )

            # If we assume n10=inf are invalid
            df = (
                self.remove_infinite_values(df, "n10")
                if self.config["preprocessing"].get("remove_inf_n10", False)
                else df
            )

            df = (
                self.cap_outliers(df, "n14", "n10", np.inf)
                if self.config["preprocessing"].get("cap_outliers", False)
                else df
            )

        df = self.normalize_engagement_metrics(df, self.engagement_features)
        df = self.drop_features(df)
        df = self.engineer_features(df)
        return df

    def normalize_engagement_metrics(
        self, df: pd.DataFrame, engagement_features: dict
    ) -> pd.DataFrame:
        """
        Normalize specified engagement metrics over the period.
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            engagement_features (dict): Keys are column names and values are periods to normalize over.
        """
        for feature, days in engagement_features.items():
            if feature in df.columns:
                df[f"{feature}_daily"] = df[feature] / days
        return df

    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not needed for further processing."""
        columns_to_drop = self.config.get("preprocessing", {}).get("drop_columns", [])
        df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            axis=1,
            inplace=True,
        )
        return df

    def filter_negative_engagement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove observations with negative engagement metrics."""
        df = df[(df["n1"] >= 0) & (df["n13"] >= 0)]
        return df

    def remove_infinite_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove rows where the specified column has infinite values."""
        logging.info("Removing infinite values for...")
        return df[df[column] != np.inf]

    def cap_outliers(
        self,
        df: pd.DataFrame,
        target_column: str,
        reference_column: str,
        reference_value,
    ) -> pd.DataFrame:
        """Cap values in target_column based on extreme values in reference_column."""

        # Apply if inf n10 are not removed
        if np.isinf(df[reference_column]).any():
            # Find the max value for capping where the reference column does not contain infinite values
            upper_limit = df[~np.isinf(df[reference_column])][target_column].max()
            # upper_limit = df[np.isinf(df[reference_column])][target_column].min()
            df[f"{target_column}_capped"] = df[target_column].clip(upper=upper_limit)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "n1" in df.columns:
            df.loc[:, "n1_squared"] = df["n1"] ** 2
        return df
