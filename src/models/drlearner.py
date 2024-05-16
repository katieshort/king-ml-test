"""
This module provides tools for creating and utilizing machine learning models
for causal inference through Double Robust Estimation techniques. It includes
utilities to construct customized model pipelines and manage the lifecycle of
causal inference models from training through evaluation to prediction.

Features include:
- The `DoubleRobustLearner` class, which encapsulates training, evaluation,
  and prediction processes using doubly robust methods to estimate causal effects
  comparing binary treatment variable.
"""

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.pipelines.pipeline_utils import create_model_pipeline
from src.utils.utils import log_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DoubleRobustLearner:
    """
    A class for Double Robust Estimation for causal inference.

    Attributes:
        propensity_model_ctor (Callable): Constructor for the propensity score model.
        outcome_model_ctor (Callable): Constructor for the outcome model.
        final_model_ctor (Callable): Constructor for the final regression model.
    """

    def __init__(
        self,
        config,
        model_type: str,
        min_propensity: float = 1e-6,
        cv_folds: int = 5,
        verbose: bool = True,
    ):
        """
        Initializes the DoubleRobustLearner with specified model constructors.

        Parameters:
            config (dict): Configuration settings for models and features.
            model_type (str): Specifies the type of model configuration to use ('engagement' or 'monetization').
            min_propensity (float): Minimum clipping value for propensity scores.
            cv_folds (int): Number of cross-validation folds for training nuisance models.
            verbose (bool): Enables detailed logging during model operations if set to True.
        """
        self.config = config
        self.model_type = model_type
        self.propensity_params = None
        self.outcome_params = None
        self.min_propensity = min_propensity
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.initialize_models()

        # Initializes storage for model details such as fitted models and their evaluation metrics
        self.model_details = {
            "propensity": {"models": [], "roc_auc": []},
            "outcome": {"models": [], "mse": [], "r2": []},
            "dr": {"mse_treated": [], "mse_control": []},
        }

    def get_model_constructor(self, path):
        """
        Dynamically import a model constructor based on its path in the configuration.

        Args:
            path (str): Dot path of the model constructor e.g., 'sklearn.linear_model.LogisticRegression'

        Returns:
            constructor: The model constructor from the specified path.
        """
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def initialize_models(self) -> None:
        """Initialize the model constructors for the propensity, outcome, and final models based on configuration."""
        if self.verbose:
            logging.info("Initializing models...")

        models_config = self.config["models"][self.model_type]

        PropensityModel = self.get_model_constructor(
            models_config["propensity_model"]["type"]
        )
        OutcomeModel = self.get_model_constructor(
            models_config["outcome_model"]["type"]
        )
        FinalModel = self.get_model_constructor(models_config["final_model"]["type"])

        self.propensity_model_ctor = create_model_pipeline(
            PropensityModel,
            models_config["propensity_model"].get("params", {}),
            self.config["features"][self.model_type]["categorical"],
            self.config["features"][self.model_type]["numerical"],
        )

        self.outcome_model_ctor = create_model_pipeline(
            OutcomeModel,
            models_config["outcome_model"].get("params", {}),
            self.config["features"][self.model_type]["categorical"],
            self.config["features"][self.model_type]["numerical"],
            models_config["outcome_model"].get("transform_target", False),
            models_config["outcome_model"].get("poly_features", []),
            models_config["outcome_model"].get("log_transform", []),
        )

        self.final_model_ctor = create_model_pipeline(
            FinalModel,
            models_config["final_model"].get("params", {}),
            [],
            [],
            models_config["final_model"].get("poly_features", []),
        )

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Extracts feature matrices and vectors from the dataframe based on configuration.

        Args:
            df (pd.DataFrame): The dataset to extract data from.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Feature matrices for X and W, treatment vector T, and outcome vector Y.
        """
        if self.verbose:
            logging.info("Preparing data...")
        X = df[self.config["features"][self.model_type]["X"]]
        W = df[self.config["features"][self.model_type]["W"]]
        T = (df[self.config["features"][self.model_type]["T"]] == "B").astype(int)
        Y = df[self.config["features"][self.model_type]["Y"]]
        return X, W, T, Y

    def cross_fit_nuisance(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs cross-fitting for nuisance models using stratified K-Fold splits.

        Parameters:
            X (ndarray): Feature matrix.
            T (ndarray): Treatment vector.
            Y (ndarray): Outcome vector.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Propensity scores, treated and control outcome predictions.
        """
        if self.verbose:
            logging.info("Performing cross-fitting...")

        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
        prop_scores = np.zeros(T.shape)
        outcome_preds_treated = np.zeros(Y.shape)
        outcome_preds_control = np.zeros(Y.shape)

        for train_index, test_index in kf.split(X, T):
            self._fit_nuisance_models(
                X.iloc[train_index],
                X.iloc[test_index],
                T.iloc[train_index],
                T.iloc[test_index],
                Y.iloc[train_index],
                Y.iloc[test_index],
                prop_scores,
                outcome_preds_treated,
                outcome_preds_control,
                test_index,
            )

        # Store propensity scores for later validation
        self.propensity_scores_df = pd.DataFrame(
            {"index": T.index, "propensity_score": prop_scores, "treatment": T}
        ).set_index("index")

        if self.verbose:
            # Log the metrics after all folds are processed
            logging.info("Cross-validation metrics per fold:")
            for i in range(self.cv_folds):
                logging.info(
                    f"Fold {i + 1}: Propensity ROC AUC = {self.model_details['propensity']['roc_auc'][i]:.4f}, "
                    f"Outcome MSE = {self.model_details['outcome']['mse'][i]:.4f}, "
                    f"Outcome R^2 = {self.model_details['outcome']['r2'][i]:.4f}"
                )

        return prop_scores, outcome_preds_treated, outcome_preds_control

    def _fit_nuisance_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.Series,
        T_test: pd.Series,
        Y_train: pd.Series,
        Y_test: pd.Series,
        prop_scores: np.ndarray,
        outcome_preds_treated: np.ndarray,
        outcome_preds_control: np.ndarray,
        test_index: np.ndarray,
    ) -> None:
        """
        Fit nuisance models (propensity and outcome models) for a cross-validation fold.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Test feature matrix.
            T_train (pd.Series): Training treatment vector.
            T_test (pd.Series): Test treatment vector.
            Y_train (pd.Series): Training outcome vector.
            Y_test (pd.Series): Test outcome vector.
            prop_scores (np.ndarray): Array to store propensity scores.
            outcome_preds_treated (np.ndarray): Array to store outcome predictions for treated.
            outcome_preds_control (np.ndarray): Array to store outcome predictions for control.
            test_index (np.ndarray): Test indices for the current fold.
        """

        propensity_model = self.train_model(
            self.propensity_model_ctor, self.propensity_params, X_train, T_train
        )
        prop_scores[test_index] = self._predict_propensity(propensity_model, X_test)
        self._update_propensity_details(
            propensity_model, T_test, prop_scores[test_index]
        )

        outcome_model = self.train_model(
            self.outcome_model_ctor,
            self.outcome_params,
            X_train.assign(treatment=T_train),  # include treatment as a feature
            Y_train,
        )
        outcome_preds_treated[test_index], outcome_preds_control[test_index] = (
            self._predict_outcomes(outcome_model, X_test)
        )
        self._update_outcome_details(outcome_model, Y_test, X_test, T_test)

        self._evaluate_and_update_dr_metrics(
            Y_test,
            T_test,
            prop_scores[test_index],
            outcome_preds_treated[test_index],
            outcome_preds_control[test_index],
        )

    def _predict_propensity(self, model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
        """Predict propensity scores and clip them."""
        prop_scores = model.predict_proba(X_test)[:, 1]
        return np.clip(prop_scores, self.min_propensity, 1 - self.min_propensity)

    def _update_propensity_details(
        self, model: Pipeline, T_test: pd.Series, prop_scores: np.ndarray
    ):
        """Update propensity model details."""
        self.model_details["propensity"]["models"].append(model)
        self.model_details["propensity"]["roc_auc"].append(
            roc_auc_score(T_test, prop_scores)
        )

    def _predict_outcomes(
        self, model: Pipeline, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict outcomes under treatment and control assumptions."""
        outcome_preds_treated = model.predict(X_test.assign(treatment=1))
        outcome_preds_control = model.predict(X_test.assign(treatment=0))
        return outcome_preds_treated, outcome_preds_control

    def _update_outcome_details(
        self,
        model: Pipeline,
        Y_test: pd.Series,
        X_test: pd.DataFrame,
        T_test: pd.Series,
    ):
        """Update outcome model details."""
        # Predict outcomes with actual treatment indicators
        outcome_predictions = model.predict(X_test.assign(treatment=T_test))
        self.model_details["outcome"]["models"].append(model)
        self.model_details["outcome"]["mse"].append(
            mean_squared_error(Y_test, outcome_predictions)
        )
        self.model_details["outcome"]["r2"].append(
            r2_score(Y_test, outcome_predictions)
        )

    def _evaluate_and_update_dr_metrics(
        self,
        Y_test: pd.Series,
        T_test: pd.Series,
        prop_scores: np.ndarray,
        outcome_preds_treated: np.ndarray,
        outcome_preds_control: np.ndarray,
    ) -> None:
        """Evaluate doubly robust estimates and update model details."""
        Y_DR_treated, Y_DR_control = self.doubly_robust_estimation(
            Y_test, T_test, prop_scores, outcome_preds_treated, outcome_preds_control
        )

        treated_indices = T_test == 1
        control_indices = T_test == 0

        if np.any(treated_indices):
            mse_treated = mean_squared_error(
                Y_test[treated_indices], Y_DR_treated[treated_indices]
            )
            self.model_details["dr"]["mse_treated"].append(mse_treated)

        if np.any(control_indices):
            mse_control = mean_squared_error(
                Y_test[control_indices], Y_DR_control[control_indices]
            )
            self.model_details["dr"]["mse_control"].append(mse_control)

    def train_model(
        self,
        model_pipeline: Callable,
        params: Optional[Dict],
        X_train: pd.DataFrame,
        Y_train: pd.Series,
    ) -> Pipeline:
        """
        Trains a model or a pipeline based on provided constructor and parameters.

        Args:
            model_pipeline (Callable): A callable that returns an sklearn pipeline or model.
            params (Dict): Parameters for hyperparameter tuning.
            X_train (pd.DataFrame): Training features.
            Y_train (pd.Series): Training target.

        Returns:
            Pipeline: Trained model pipeline.
        """

        model = model_pipeline()
        if params:
            model = RandomizedSearchCV(model, params, cv=3)
        model.fit(X_train, Y_train)
        return model

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate and return the mean and standard deviation of evaluation metrics.

        Returns:
            Dict[str, Dict[str, float]]: Summary statistics for propensity and outcome models.
        """
        propensity_metrics = self._calculate_metrics(
            self.model_details["propensity"]["roc_auc"]
        )
        outcome_mse_metrics = self._calculate_metrics(
            self.model_details["outcome"]["mse"]
        )
        outcome_r2_metrics = self._calculate_metrics(
            self.model_details["outcome"]["r2"]
        )

        metrics = {
            "propensity": {
                "roc_auc": propensity_metrics,
            },
            "outcome": {
                "mse": outcome_mse_metrics,
                "r2": outcome_r2_metrics,
            },
        }

        if self.verbose:
            # Log using utility
            log_metrics(metrics)

        return metrics

    def _calculate_metrics(self, values: List[float]) -> Dict[str, float]:
        """Calculate mean and standard deviation for a list of values."""
        return {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    def doubly_robust_estimation(
        self,
        Y: pd.Series,
        T: pd.Series,
        prop_scores: np.ndarray,
        outcome_preds_treated: np.ndarray,
        outcome_preds_control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates doubly robust estimates for both treated and control groups.

        Args:
            Y (pd.Series): Outcome vector.
            T (pd.Series): Treatment vector.
            prop_scores (np.ndarray): Propensity scores.
            outcome_preds_treated (np.ndarray): Predicted outcomes under treatment.
            outcome_preds_control (np.ndarray): Predicted outcomes under control.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Doubly robust estimates for treated and control potential outcomes.
        """
        treated = T == 1
        control = ~treated

        Y_DR_treated = (
            outcome_preds_treated + (Y - outcome_preds_treated) / prop_scores * treated
        )
        Y_DR_control = (
            outcome_preds_control
            + (Y - outcome_preds_control) / (1 - prop_scores) * control
        )

        return Y_DR_treated, Y_DR_control

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the DoubleRobustLearner model using provided data.

        Args:
            df (pd.DataFrame): Dataframe containing feature and target data.
        """

        X, W, T, Y = self.prepare_data(df)
        XW = pd.concat([X, W], axis=1)  # Combine X and W for the nuisance models

        prop_scores, outcome_preds_treated, outcome_preds_control = (
            self.cross_fit_nuisance(XW, T, Y)
        )
        Y_DR_treated, Y_DR_control = self.doubly_robust_estimation(
            Y, T, prop_scores, outcome_preds_treated, outcome_preds_control
        )

        delta_Y_DR = Y_DR_treated - Y_DR_control
        if self.verbose:
            logging.info("Training final model...")
        self.final_model = self.final_model_ctor()
        self.final_model.fit(X, delta_Y_DR)

    def predict_ate(self, df) -> float:
        """
        Predicts the Average Treatment Effect (ATE) for the given features.

        Args:
            df (pd.DataFrame): Dataframe containing feature data.

        Returns:
            float: The estimated ATE.
        """
        X, _, _, _ = self.prepare_data(df)
        cate_estimates = self.final_model.predict(X)
        return np.mean(cate_estimates)

    def predict_cate(self, df) -> np.ndarray:
        """
        Predicts Conditional Average Treatment Effects (CATE) for new data.

        Args:
            df (pd.DataFrame): Dataframe containing new feature matrix.

        Returns:
            np.ndarray: Predicted treatment effects.
        """
        X, _, _, _ = self.prepare_data(df)
        return self.final_model.predict(X)

    def get_model_details(self) -> Dict[str, Any]:
        """
        Retrieves the model coefficients, intercept, and possibly statistical details.
        """
        final_estimator = self.final_model.named_steps["model"]
        if hasattr(final_estimator, "coef_"):
            return {
                "intercept": final_estimator.intercept_,
                "coefficients": final_estimator.coef_,
            }
        return {"error": "Model details not available."}

    def get_propensity_scores_df(self) -> pd.DataFrame:
        """
        Retrieve the propensity scores DataFrame after training.
        """
        return self.propensity_scores_df

    def save_model(self) -> None:
        """
        Save the trained final model to a file.
        """
        model_path = self.config["file_paths"]["model_path"][self.model_type]
        joblib.dump(self.final_model, model_path)
        if self.verbose:
            logging.info(f"Model saved to {model_path}")

    def load_model(self) -> None:
        """
        Load model from a specified file.
        """
        model_path = self.config["file_paths"]["model_path"][self.model_type]
        self.final_model = joblib.load(model_path)
        if self.verbose:
            logging.info(f"Model loaded from {model_path}")
