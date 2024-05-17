import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_design_distribution(df, save_path=None):
    """
    Plot the distribution of actual vs. recommended design.

    Args:
    df_results (pd.DataFrame): DataFrame containing columns 'player_group' and 'recommended_design'.
    """
    actual_design_counts = df["player_group"].value_counts()
    recommended_design_counts = df["recommended_design"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(2)

    bar1 = ax.bar(
        index, actual_design_counts, bar_width, label="Actual Design", color="b"
    )
    bar2 = ax.bar(
        index + bar_width,
        recommended_design_counts,
        bar_width,
        label="Recommended Design",
        color="r",
    )

    plt.xlabel("Group")
    plt.ylabel("Number of Players")
    plt.title("Distribution of Actual vs. Recommended Design")
    plt.xticks(index + bar_width / 2, ("Design A", "Design B"))
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def overlay_design_distributions(df, save_path=None):
    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Define the color palette for clear differentiation
    palette = {
        "Original A": "blue",
        "Original B": "orange",
        "Recommended A": "lightblue",
        "Recommended B": "peachpuff",
    }

    # Adding 'Original' prefix to values in 'player_group'
    df["original_design"] = df["player_group"].dropna().apply(lambda x: f"Original {x}")

    # Plotting original group assignments
    sns.histplot(
        data=df,
        x="n1_daily",
        hue="original_design",
        element="step",
        stat="count",
        common_norm=False,
        fill=False,
        palette={key: palette[key] for key in ["Original A", "Original B"]},
        linewidth=2,
        alpha=0.8,
    )

    # Adding 'Recommended' prefix to values in 'recommended_design'
    df["modified_design"] = (
        df["recommended_design"].dropna().apply(lambda x: f"Recommended {x}")
    )

    # Plotting recommended design distributions
    sns.histplot(
        data=df,
        x="n1_daily",
        hue="modified_design",
        element="step",
        stat="count",
        common_norm=False,
        fill=True,
        alpha=0.5,
        palette={key: palette[key] for key in ["Recommended A", "Recommended B"]},
    )

    plt.title("Comparison of Original and Recommended Design Distributions")
    plt.xlabel("n1_daily")
    plt.ylabel("Count")
    plt.yscale("log")

    # Extract handles and labels and then modify them for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Manually adjusting legend to reflect correct labels and colors
    labels = ["Original A", "Original B", "Recommended A", "Recommended B"]
    colors = [palette[label] for label in labels]
    new_handles = [
        plt.Line2D([], [], color=color, linestyle="-", linewidth=2) for color in colors
    ]
    plt.legend(new_handles, labels, title="Group")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_distributions_per_group(df):

    for group in ["A", "B"]:
        sns.histplot(
            data=df[df["player_group"] == group],
            x="n1_daily",
            hue="recommended_design",
        )
        plt.title(f"Design Distribution for Group {group}")
        plt.show()


def plot_propensity_scores(prop_scores, treatment):
    # Normalize by the number of observations in each group
    n_treated = (treatment == 1).sum()
    n_control = (treatment == 0).sum()

    plt.figure(figsize=(10, 6))
    plt.hist(
        prop_scores[treatment == 1],
        bins=100,
        alpha=0.5,
        label="Design B",
        density=False,
        weights=np.ones(n_treated) / n_treated,
    )
    plt.hist(
        prop_scores[treatment == 0],
        bins=100,
        alpha=0.5,
        label="Design A",
        density=False,
        weights=np.ones(n_control) / n_control,
    )
    plt.xlabel("Propensity Score")
    plt.ylabel("Frequency")
    plt.legend(loc="best")
    plt.title("Distribution of Propensity Scores")
    plt.show()


def plot_propensity_histograms(prop_scores_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=prop_scores_df,
        x="propensity_score",
        hue="treatment",
        bins=60,
        kde=False,
        element="step",
        stat="density",
        common_norm=True,
    )
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of Propensity Scores")
    plt.show()


def plot_density_propensity_scores(prop_scores, treatment, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(prop_scores[treatment == 1], fill=True, color="blue", label="Design B")
    sns.kdeplot(
        prop_scores[treatment == 0], fill=True, color="orange", label="Design A"
    )
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.title("Density Plot of Propensity Scores")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compare_design_distributions(df):

    # Set up a larger figure size for readability
    plt.figure(figsize=(14, 6))

    # Plotting the original group assignments
    plt.subplot(1, 2, 1)  # 1 row, 2 cols, 1st subplot
    sns.histplot(
        data=df,
        x="n1_daily",
        hue="player_group",
        element="step",
        stat="count",
        common_norm=False,
        palette="deep",
    )
    plt.title("Original Design Assignment by Group")
    plt.xlabel("n1")
    plt.ylabel("Count")
    plt.yscale("log")

    # Plotting the recommended design distributions
    plt.subplot(1, 2, 2)  # 1 row, 2 cols, 2nd subplot
    sns.histplot(
        data=df,
        x="n1_daily",
        hue="recommended_design",
        element="step",
        stat="count",
        common_norm=False,
        palette="pastel",
    )
    plt.title("Recommended Design Distribution")
    plt.xlabel("n1")
    plt.ylabel("Count")
    plt.yscale("log")

    plt.tight_layout()
    plt.show()
