import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_eda():

    data = load_data()

    charts = []

    sns.set_style("whitegrid")

    # Missing Values
    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10,5))
        sns.barplot(x=missing.index, y=missing.values)

        plt.xticks(rotation=45)
        plt.title("Missing Values")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_values.png"))
        plt.close()

        charts.append("missing_values.png")

        plt.figure(figsize=(12,6))
        sns.heatmap(data.isnull(), cbar=False)

        plt.title("Missing Value Heatmap")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_heatmap.png"))
        plt.close()

        charts.append("missing_heatmap.png")


    # -----------------------------
    # Basic Information
    # -----------------------------
    duplicates = int(data.duplicated().sum())

    target_counts = {}

    if "PlacementStatus" in data.columns:
        target_counts = data["PlacementStatus"].value_counts().to_dict()

        plt.figure(figsize=(6, 5))

        sns.countplot(
            x="PlacementStatus",
            data=data
        )

        plt.title("Placement Status Distribution")

        plt.tight_layout()

        plt.savefig(_chart_path("placement_status.png"))
        plt.close()

        charts.append("placement_status.png")

    # -----------------------------
    # Histograms
    # -----------------------------
    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:

        data[hist_cols].hist(
            figsize=(14, 10),
            bins=20
        )

        plt.tight_layout()
        plt.savefig(_chart_path("numeric_distribution.png"))
        plt.close()

        charts.append("numeric_distribution.png")

    # -----------------------------
    # CGPA Distribution
    # -----------------------------
    if "CGPA" in data.columns:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            data["CGPA"],
            kde=True
        )

        plt.axvline(
            data["CGPA"].mean(),
            color="red",
            linestyle="--",
            label="Mean"
        )

        plt.legend()

        plt.title("CGPA Distribution")

        plt.tight_layout()

        plt.savefig(_chart_path("cgpa_distribution.png"))
        plt.close()

        charts.append("cgpa_distribution.png")

    # -----------------------------
    # Boxplots
    # -----------------------------
    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=data[col],
            color="skyblue"
        )

        plt.title(col)

        plt.tight_layout()

        plt.savefig(_chart_path(f"boxplot{col}.png"))
        plt.close()

        charts.append(f"boxplot_{col}.png")

    # -----------------------------
    # CGPA vs Salary
    # -----------------------------
    if "CGPA" in data.columns and "SalaryPackage" in data.columns:

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=data,
            x="CGPA",
            y="SalaryPackage",
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"}
        )

        plt.title("CGPA vs Salary")

        plt.tight_layout()

        plt.savefig(_chart_path("cgpa_salary.png"))
        plt.close()

        charts.append("cgpa_salary.png")

    # -----------------------------
    # Aptitude vs Coding
    # -----------------------------
    if (
        "AptitudeTestScore" in data.columns and
        "CodingTestScore" in data.columns
    ):

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=data,
            x="AptitudeTestScore",
            y="CodingTestScore"
        )

        plt.title("Aptitude vs Coding")

        plt.tight_layout()

        plt.savefig(_chart_path("aptitude_coding.png"))
        plt.close()

        charts.append("aptitude_coding.png")
        # -----------------------------
        # Categorical Count Plots
        # -----------------------------
        cat_cols = [
            "Gender",
            "City",
            "CollegeTier",
            "Stream",
            "Specialisation",
            "Hostel",
            "HistoryOfBacklogs",
            "CGPA_Tier"
        ]

        cat_cols = [c for c in cat_cols if c in data.columns]

        for col in cat_cols:
            plt.figure(figsize=(10, 5))

            sns.countplot(
                data=data,
                x=col,
                order=data[col].value_counts().index
            )

            plt.xticks(rotation=45)
            plt.title(f"{col} Distribution")

            plt.tight_layout()

            plt.savefig(_chart_path(f"{col}.png"))
            plt.close()

            charts.append(f"{col}.png")

        # -----------------------------
        # Gender vs Placement
        # -----------------------------
        if (
                "Gender" in data.columns and
                "PlacementStatus" in data.columns
        ):
            plt.figure(figsize=(7, 5))

            sns.countplot(
                data=data,
                x="Gender",
                hue="PlacementStatus"
            )

            plt.title("Gender vs Placement Status")

            plt.tight_layout()

            plt.savefig(_chart_path("gender_placement.png"))
            plt.close()

            charts.append("gender_placement.png")

        # -----------------------------
        # College Tier vs Placement
        # -----------------------------
        if (
                "CollegeTier" in data.columns and
                "PlacementStatus" in data.columns
        ):
            plt.figure(figsize=(7, 5))

            sns.countplot(
                data=data,
                x="CollegeTier",
                hue="PlacementStatus"
            )

            plt.title("College Tier vs Placement")

            plt.tight_layout()

            plt.savefig(_chart_path("college_tier.png"))
            plt.close()

            charts.append("college_tier.png")

        # -----------------------------
        # Pair Plot
        # -----------------------------
        pair_cols = [
            "CGPA",
            "AttendancePercent",
            "CodingTestScore",
            "PlacementStatus"
        ]

        pair_cols = [c for c in pair_cols if c in data.columns]

        if (
                len(pair_cols) > 1 and
                "PlacementStatus" in pair_cols
        ):
            pair = sns.pairplot(
                data=data[pair_cols],
                hue="PlacementStatus"
            )

            pair.savefig(_chart_path("pairplot.png"))

            plt.close("all")

            charts.append("pairplot.png")

        # -----------------------------
        # Return Results
        # -----------------------------
        return {
            "rows": len(data),
            "columns": len(data.columns),
            "duplicates": duplicates,
            "missing": missing.to_dict(),
            "target_counts": target_counts,
            "charts": charts
        }