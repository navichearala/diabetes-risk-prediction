"""
Exploratory Data Analysis for the Diabetes Readmission Prediction project.

Generates and saves visualizations: target distribution, correlation heatmap,
feature distributions by outcome, BMI vs Glucose risk scatter, and ROC-ready stats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CONFIG, resolve
from data_preprocessing import clean_data, load_data

sns.set_theme(style="whitegrid", palette="muted")

FIG_DIR = resolve(CONFIG["paths"]["figures_dir"])
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Outcome"].value_counts().sort_index()
    ax.bar(["No Diabetes", "Diabetes"], counts.values, color=["#4C9F70", "#D7263D"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Target Distribution: Diabetes Outcome")
    ax.set_ylabel("Patient count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_distribution.png", dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=120)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "DiabetesPedigreeFunction"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), features):
        sns.kdeplot(data=df, x=feat, hue="Outcome", fill=True, ax=ax, common_norm=False)
        ax.set_title(f"{feat} by Outcome")
    fig.suptitle("Feature Distributions by Diabetes Outcome", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_feature_distributions.png", dpi=120)
    plt.close(fig)


def plot_glucose_bmi_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Glucose",
        y="BMI",
        hue="Outcome",
        palette={0: "#4C9F70", 1: "#D7263D"},
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(125, ls="--", color="grey", label="Diabetic glucose threshold")
    ax.axhline(30, ls="--", color="orange", label="Obesity BMI threshold")
    ax.set_title("Glucose vs BMI by Outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_glucose_bmi_scatter.png", dpi=120)
    plt.close(fig)


def run_eda(data_path: str | Path) -> None:
    df = clean_data(load_data(data_path))
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_glucose_bmi_scatter(df)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_eda(resolve(CONFIG["data"]["raw_path"]))
