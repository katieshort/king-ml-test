from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Any, Optional, List

def create_model_pipeline(
    model_constructor: Callable,
    categorical_features: list,
    numerical_features: list,
    transform_target: bool = False,
    poly_features: Optional[List[str]] = None,
    scale: bool = False,
    log_transform: Optional[List[str]] = None,
) -> Callable:

    """Factory function to create and return a pipeline constructor function."""

    def build_pipeline():
        # Individual transformers applied directly within ColumnTransformer
        transformers = []

        # Adding a simple imputer for all numerical features
        transformers.append(("imputer", SimpleImputer(strategy="mean"), numerical_features))

        # Apply log transformation directly if specified
        if log_transform:
            log_transformer = FunctionTransformer(np.log1p, validate=False)
            transformers.append(("log_transform", log_transformer, log_transform))

        # Apply polynomial features directly if specified
        if poly_features:
            poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
            transformers.append(("poly_transform", poly_transformer, poly_features))

        # Standard scaling for numerical features if specified
        if scale:
            scaler = StandardScaler()
            transformers.append(("scaler", scaler, numerical_features))

        # Categorical features encoding
        if categorical_features:
            cat_transformer = TargetEncoder(smoothing=1.0, handle_missing="value")
            transformers.append(("cat_transform", cat_transformer, categorical_features))

        # Combine all transformations
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough"  # keep all other features unchanged
        )

        # Configure the model, with optional target transformation
        model = (TransformedTargetRegressor(
                    regressor=model_constructor(),
                    func=np.log1p,
                    inverse_func=np.expm1)
                 if transform_target
                 else model_constructor())

        # The final pipeline including preprocessing and the model itself
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    return build_pipeline
