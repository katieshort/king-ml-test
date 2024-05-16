"""
Module: dataloader.py

This module provides the DataLoader class responsible for loading and preprocessing data, functionalities for reading data from CSV files, preprocessing it through various transformations such as normalization, outlier capping, and feature engineering, and preparing it for downstream model training and inference.

The module is designed to be flexible, allowing for easy adjustments to data paths and preprocessing steps through configuration.

Classes:
    DataLoader: Loads and preprocesses data from configured sources.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """
    Data loader class for handling data ingestion and preprocessing.

    Attributes:
        config (dict): Configuration dictionary containing file paths and preprocessing parameters.
        verbose (bool): Flag to enable verbose output. Defaults to False.
        train (bool): Flag to distinguish between training or testing mode. Defaults to True.
        engagement_features (dict): Dictionary specifying features and periods for normalization.
    """

    def __init__(self, config: Dict, verbose: bool = False, train: bool = True):
        """Initialize the DataLoader with configuration, verbosity level and mode."""
        self.config = config
        self.verbose = verbose
        self.train = train
        self.engagement_features = self.config.get("preprocessing", {}).get(
            "engagement_features", {"n1": 30, "n13": 7}
        )

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a specified path in config.

        Attempts to read a CSV file from the path provided in the 'data_path' configuration.
        If any error occurs during loading, it will log the error and re-raise the exception.

        Returns:
            pd.DataFrame: A dataframe loaded from the specified file path.

        Raises:
            Exception: Re-raises any exception that occurs during data loading.
        """
        try:
            df = pd.read_csv(self.config.get("file_paths", {}).get("data_path"))
            logging.info(
                "Data loaded successfully from %s",
                self.config["file_paths"]["data_path"],
            )
            return df
        except Exception as e:
            logging.error("Failed to load data: %s", e)
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by applying filters, normalization, and feature engineering.

        Args:
            df (pd.DataFrame): Dataframe to preprocess.
        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        df = df.copy()

        if self.train:
            df = (
                self.filter_negative_engagement(df)
                if self.config["preprocessing"].get("filter_negative_engagement", False)
                else df
            )

            df = (
                self.remove_infinite_values(df, "n10")
                if self.config["preprocessing"].get("remove_inf_n10", False)
                else df
            )

            df = (
                self.cap_outliers(df, "n14", "n10")
                if self.config["preprocessing"].get("cap_outliers", False)
                else df
            )

        df = self.normalize_engagement_metrics(df)
        df = self.drop_features(df)
        df = self.engineer_features(df)
        return df

    def normalize_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize specified engagement metrics over the period.
        Args:
            df (pd.DataFrame): Dataframe containing the data.
        Returns:
            pd.DataFrame: Dataframe with normalized metrics.
        """
        for feature, days in self.engagement_features.items():
            if feature in df.columns:
                df[f"{feature}_daily"] = df[feature] / days
        return df

    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns not needed based on configuration.

        Args:
            df (pd.DataFrame): Dataframe from which to drop columns.

        Returns:
            pd.DataFrame: Dataframe with unnecessary columns dropped.
        """
        columns_to_drop = self.config.get("preprocessing", {}).get("drop_columns", [])
        df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            axis=1,
            inplace=True,
        )
        return df

    def filter_negative_engagement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove observations with negative engagement metrics.

        Args:
            df (pd.DataFrame): Dataframe to filter.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        return df[(df["n1"] >= 0) & (df["n13"] >= 0)]

    def remove_infinite_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove rows where the specified column has infinite values.

        Args:
            df (pd.DataFrame): Dataframe to filter.
            column (str): Column name to check for infinite values.

        Returns:
            pd.DataFrame: Dataframe with infinite values removed.
        """
        return df[df[column] != np.inf]

    def cap_outliers(
        self, df: pd.DataFrame, target_column: str, reference_column: str
    ) -> pd.DataFrame:
        """
        Cap values in target_column based on extreme values in reference_column.

        This function caps the values in `target_column` by finding the maximum valid value in the
        `target_column` that doesn't correspond to an infinite value in the `reference_column`.

        Args:
            df (pd.DataFrame): Dataframe containing the data.
            target_column (str): The column in which values will be capped.
            reference_column (str): The column used to determine if values in target_column should be capped.

        Returns:
            pd.DataFrame: The modified dataframe with capped values in `target_column`.
        """

        # Check for and handle infinite values in the reference column
        if np.isinf(df[reference_column]).any():
            # Find the maximum value in the target column where the reference column is not infinite
            upper_limit = df[~np.isinf(df[reference_column])][target_column].max()
            # Apply capping to the target column
            df[f"{target_column}_capped"] = df[target_column].clip(upper=upper_limit)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features based on existing data in the dataframe.

        This function currently creates a squared feature of 'n1' if it exists. This is a starting point,
        and the feature engineering process could be expanded significantly with more complex transformations
        and the creation of new features based on domain knowledge and model requirements.

        Args:
            df (pd.DataFrame): Dataframe containing the data to be transformed.

        Returns:
            pd.DataFrame: The dataframe with new engineered features.
        """
        if "n1" in df.columns:
            df.loc[:, "n1_squared"] = df["n1"] ** 2

        # Example placeholder for additional feature engineering
        # This could include interaction terms, polynomial features, or domain-specific transformations
        # Example: df['new_feature'] = df['existing_feature1'] * df['existing_feature2']

        return df
