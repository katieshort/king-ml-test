import yaml
import numpy as np
from prettytable import PrettyTable
import logging


def load_config(path: str):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def calculate_smd(df, treatment_col, covariate_cols):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    smd_list = {}
    for col in covariate_cols:
        mean_treated = treated[col].mean()
        mean_control = control[col].mean()
        pooled_sd = np.sqrt((treated[col].std() ** 2 + control[col].std() ** 2) / 2)
        smd = np.abs(mean_treated - mean_control) / pooled_sd
        smd_list[col] = smd
    return smd_list


def log_metrics(details):
    table = PrettyTable()
    table.field_names = ["Component", "Metric", "Mean", "Std Dev"]

    # Iterate over the components and their metrics in the provided dictionary
    for component, metrics in details.items():
        for metric, values in metrics.items():
            # Format mean and std dev in scientific notation
            mean = f"{values['mean']:.4e}"  # Using .2e to show two digits after the decimal in scientific notation
            std = f"{values['std']:.2e}"  # Same here for consistency
            table.add_row(
                [
                    component.capitalize(),  # Component name, e.g., 'Propensity', 'Outcome'
                    metric.upper(),  # Metric name, e.g., 'ROC_AUC', 'MSE', 'R2'
                    mean,  # Mean value in scientific notation
                    std,  # Standard deviation in scientific notation
                ]
            )

    # Log the table as an info message
    logging.info("\n" + str(table))