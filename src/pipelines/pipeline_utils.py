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



# class CustomTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, log_features=None, poly_features=None):
#         self.log_features = log_features
#         self.poly_features = poly_features
#
#     def fit(self, X, y=None):
#         if self.poly_features:
#             # Initialize the PolynomialFeatures transformer to fit later
#             self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
#             self.poly_transformer.fit(X[self.poly_features])
#         return self
#
#     def transform(self, X):
#         X = X.copy()
#         if self.log_features:
#             X.loc[:, self.log_features] = np.log1p(X.loc[:, self.log_features])
#
#         if self.poly_features:
#             poly_data = self.poly_transformer.transform(X[self.poly_features])
#             # Get the original names to create meaningful interaction terms
#             input_features = [
#                 X.columns[X.columns.get_loc(f)] for f in self.poly_features
#             ]
#             # Create descriptive names for the polynomial features
#             feature_names = self.poly_transformer.get_feature_names(
#                 input_features=input_features
#             )
#             # Create a DataFrame for the polynomial features
#             poly_df = pd.DataFrame(poly_data, columns=feature_names, index=X.index)
#             # Concatenate the polynomial features DataFrame with the original DataFrame
#             X = pd.concat([X, poly_df], axis=1)
#         return X
#
#
# def create_model_pipeline(
#     model_constructor: Callable,
#     categorical_features: list,
#     numerical_features: list,
#     transform_target: bool = False,
#     poly_features: Optional[List[str]] = None,
#     scale: bool = False,
#     log_transform: Optional[List[str]] = None,
# ) -> Callable:
#
#     """Factory function to create and return a pipeline constructor function."""
#     def build_pipeline():
#
#         numeric_transformer = Pipeline(
#             [
#                 ("imputer", SimpleImputer(strategy="mean")),
#                 ("custom_transforms", CustomTransformer(log_transform, poly_features)),
#                 (
#                     ("scaler", StandardScaler())
#                     if scale
#                     else ("pass", FunctionTransformer(validate=False))
#                 ),
#             ]
#         )
#
#         categorical_transformer = Pipeline(
#             [
#                 ("target_encoder", TargetEncoder(smoothing=1.0, handle_missing="value")),
#             ]
#         )
#
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numerical_features),
#                 ("cat", categorical_transformer, categorical_features),
#             ],
#             remainder="passthrough",
#         )
#
#         model = (
#             TransformedTargetRegressor(
#                 regressor=model_constructor(), func=np.log1p, inverse_func=np.expm1
#             )
#             if transform_target
#             else model_constructor()
#         )
#
#         return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
#
#
#     return build_pipeline



# def create_model_pipeline(
#     model_constructor: Callable,
#     categorical_features: list,
#     numerical_features: list,
#     transform_target: bool = False,
#     poly_features: Optional[List[str]] = None,
#     scale: bool = False,
#     log_transform: Optional[List[str]] = None,
# ) -> Callable:
#     """Create and return a pipeline factory for the specified model configuration.
#
#     Args:
#         model_constructor (Callable): The model constructor function.
#         categorical_features (list): List of names for categorical features.
#         numerical_features (list): List of names for numerical features.
#         transform_target (bool): Whether to transform the target variable.
#         poly (bool): Whether to include polynomial features.
#         scale (bool): Whether to scale features.
#
#     Returns:
#         Callable: A function to create the configured pipeline.
#     """
#
#     def pipeline() -> Pipeline:
#
#         print("Log Transform in pipeline:", log_transform)  # Debugging output
#         print("Poly Features in pipeline:", poly_features)  # Debugging output
#         numeric_steps = [("imputer", SimpleImputer(strategy="mean"))]
#
#         if log_transform:
#             # Apply log transformation only to specified features
#             log_transformer = ColumnTransformer(
#                 [("log", FunctionTransformer(np.log1p, validate=False), log_transform)],
#                 remainder="passthrough",
#             )
#             numeric_steps.append(("log", log_transformer))
#
#         if poly_features:
#             poly_step = ColumnTransformer(
#                 [
#                     (
#                         "poly_transform",
#                         PolynomialFeatures(degree=2, include_bias=False),
#                         poly_features,
#                     )
#                 ],
#                 remainder="passthrough",
#             )
#             numeric_steps.append(("poly", poly_step))
#
#         if scale:
#             numeric_steps.append(("scaler", StandardScaler()))
#
#         numeric_transformer = Pipeline(steps=numeric_steps)
#
#         categorical_transformer = Pipeline(
#             steps=[
#                 (
#                     "target_encoder",
#                     TargetEncoder(
#                         smoothing=1.0,
#                         handle_missing="value",  # missing categories are encoded with target mean
#                     ),  # Parameter can be tuned later
#                 ),
#             ]
#         )
#
#         # Define a ColumnTransformer that applies transformations to specified columns
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numerical_features),
#                 ("cat", categorical_transformer, categorical_features),
#             ],
#             remainder="passthrough",  # Non-specified columns are passed through unchanged
#         )
#
#         model = (
#             TransformedTargetRegressor(
#                 regressor=model_constructor(), func=np.log1p, inverse_func=np.expm1
#             )
#             if transform_target
#             else model_constructor()
#         )
#
#         # Create and return the complete pipeline
#         return Pipeline(
#             steps=[
#                 ("preprocessor", preprocessor),
#                 ("model", model),
#             ]
#         )
#
#     return pipeline


# def create_model_pipeline(
#     model_constructor: Callable,
#     categorical_features: list,
#     numerical_features: list,
#     transform_target: bool = False,
#     poly_features: Optional[List[str]] = None,
#     scale: bool = False,
#     log_transform: Optional[List[str]] = None,
# ) -> Callable:
#     """Create and return a pipeline factory for the specified model configuration."""
#
#     def pipeline() -> Pipeline:
#         # Initialize the steps for numerical features
#         numeric_steps = [("imputer", SimpleImputer(strategy="mean"))]
#
#         # Log transformation
#         if log_transform:
#             # Apply log transformation only to specified features
#             numeric_steps.append(
#                 (
#                     "log_transformer",
#                     ColumnTransformer(
#                         [
#                             (
#                                 "log",
#                                 FunctionTransformer(np.log1p, validate=False),
#                                 log_transform,
#                             )
#                         ],
#                         remainder="passthrough",
#                     ),
#                 )
#             )
#
#         # Polynomial features
#         if poly_features:
#             # Apply polynomial transformation only to specified features
#             numeric_steps.append(
#                 (
#                     "poly_transformer",
#                     ColumnTransformer(
#                         [
#                             (
#                                 "poly",
#                                 PolynomialFeatures(degree=2, include_bias=False),
#                                 poly_features,
#                             )
#                         ],
#                         remainder="passthrough",
#                     ),
#                 )
#             )
#
#         # Scaling
#         if scale:
#             numeric_steps.append(("scaler", StandardScaler()))
#
#         # Assemble the numeric transformer pipeline
#         numeric_transformer = Pipeline(steps=numeric_steps)
#
#         # Categorical features transformation
#         categorical_transformer = Pipeline(
#             steps=[
#                 (
#                     "target_encoder",
#                     TargetEncoder(smoothing=1.0, handle_missing="value"),
#                 ),
#             ]
#         )
#
#         # Main preprocessor that combines numerical and categorical transformations
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numerical_features),
#                 ("cat", categorical_transformer, categorical_features),
#             ],
#             remainder="passthrough",
#         )
#
#         # Model with optional target transformation
#         model = (
#             TransformedTargetRegressor(
#                 regressor=model_constructor(), func=np.log1p, inverse_func=np.expm1
#             )
#             if transform_target
#             else model_constructor()
#         )
#
#         # Complete pipeline
#         return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
#
#     return pipeline