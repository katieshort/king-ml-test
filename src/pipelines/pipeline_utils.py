"""
This module provides a factory function for configuring and constructing machine learning pipelines.

The `create_model_pipeline` function returns a constructor function that, when called, builds a
comprehensive sklearn pipeline integrating preprocessing and modeling steps. This setup is designed
to facilitate the easy configuration and instantiation of pipelines with variable preprocessing
options tailored to different modeling needs.
"""

from typing import Callable, List, Optional

import numpy as np
from category_encoders import TargetEncoder, WOEEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)


def create_model_pipeline(
    model_constructor: Callable,
    params: dict,
    categorical_features: list,
    numerical_features: list,
    transform_target: bool = False,
    poly_features: Optional[List[str]] = None,
    scale: bool = False,
    log_transform: Optional[List[str]] = None,
    encoding_strategy: str = "target",
) -> Callable:
    """Factory function to create and return a pipeline constructor function."""

    def build_pipeline():
        transformers = []

        # Adding a simple imputer for all numerical features
        transformers.append(
            ("imputer", SimpleImputer(strategy="mean"), numerical_features)
        )

        # Apply log transformation if specified
        if log_transform:
            log_transformer = FunctionTransformer(np.log1p, validate=False)
            transformers.append(("log_transform", log_transformer, log_transform))

        # Apply polynomial features if specified
        if poly_features:
            poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformers.append(("poly_transform", poly_transformer, poly_features))

        # Standard scaling for numerical features if specified
        if scale:
            scaler = StandardScaler()
            transformers.append(("scaler", scaler, numerical_features))

        # Categorical features encoding based on the specified strategy
        if categorical_features:
            if encoding_strategy == "target":
                cat_transformer = TargetEncoder(
                    smoothing=2.0, handle_missing="value"
                )
            elif encoding_strategy == "woe":
                cat_transformer = WOEEncoder(randomized=False)
            else:
                raise ValueError(f"Unknown encoding strategy: {encoding_strategy}")

            transformers.append(
                ("cat_transform", cat_transformer, categorical_features)
            )

        # Combine all transformations
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",  # keep all other features unchanged
        )

        # Configure the model, with optional target transformation
        model = (
            TransformedTargetRegressor(
                regressor=model_constructor(**params),
                func=np.log1p,
                inverse_func=np.expm1,
            )
            if transform_target
            else model_constructor(**params)
        )

        # The final pipeline including preprocessing and the model itself
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    return build_pipeline
