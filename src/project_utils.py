"""Reusable data and presentation helpers for the project notebooks."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


REQUIRED_COLUMNS = [
    "Time",
    *[f"V{number}" for number in range(1, 29)],
    "Amount",
    "Class",
]


def create_directory_structure(base_dir: str | Path = ".") -> None:
    """Create the directories used by the project without deleting existing work."""
    root = Path(base_dir)
    for relative_path in (
        "data",
        "src",
        "results",
        "images/deliverable_1",
        "images/deliverable_2",
        "images/deliverable_3",
        "images/final",
        "rubrics",
        "report",
        "presentation",
    ):
        (root / relative_path).mkdir(parents=True, exist_ok=True)


def find_dataset_path(base_dir: str | Path = ".") -> Path:
    """Return the first approved dataset location or raise clear setup guidance."""
    root = Path(base_dir)
    candidates = (root / "data" / "creditcard.csv", root / "creditcard.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Credit Card Fraud Detection dataset not found. Download creditcard.csv "
        "from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place "
        "it at data/creditcard.csv. Do not commit the dataset."
    )


def load_credit_card_data(base_dir: str | Path = ".") -> tuple[pd.DataFrame, Path]:
    """Load the approved Kaggle dataset and verify its required schema."""
    dataset_path = find_dataset_path(base_dir)
    dataframe = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    unexpected_columns = [column for column in dataframe.columns if column not in REQUIRED_COLUMNS]
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema does not match the approved credit-card dataset. "
            f"Missing columns: {missing_columns}; unexpected columns: {unexpected_columns}."
        )
    return dataframe, dataset_path


def clean_credit_card_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate core fields and remove only exact duplicate transaction records."""
    cleaned = dataframe.copy()
    if cleaned["Class"].isna().any():
        raise ValueError("Class contains missing values and cannot be used safely as a target.")
    if not set(cleaned["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 for legitimate and 1 for fraudulent.")
    if (cleaned["Time"] < 0).any() or (cleaned["Amount"] < 0).any():
        raise ValueError("Time and Amount must be nonnegative in the approved dataset.")
    return cleaned.drop_duplicates().reset_index(drop=True)


def create_base_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add non-destructive time and amount features used in exploration."""
    featured = dataframe.copy()
    featured["Hour"] = (featured["Time"] / 3600) % 24
    featured["DayIndex"] = np.floor(featured["Time"] / 86400).astype(int)
    featured["LogAmount"] = np.log1p(featured["Amount"])
    return featured


def create_cleaning_summary(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize exact-duplicate removal and its effect on both target classes."""
    original_counts = original["Class"].value_counts().reindex([0, 1], fill_value=0)
    cleaned_counts = cleaned["Class"].value_counts().reindex([0, 1], fill_value=0)
    rows = [
        ("Original rows", len(original)),
        ("Exact duplicate rows", int(original.duplicated().sum())),
        ("Rows after duplicate removal", len(cleaned)),
        ("Original legitimate rows", int(original_counts.loc[0])),
        ("Original fraud rows", int(original_counts.loc[1])),
        ("Cleaned legitimate rows", int(cleaned_counts.loc[0])),
        ("Cleaned fraud rows", int(cleaned_counts.loc[1])),
        ("Legitimate duplicates removed", int(original_counts.loc[0] - cleaned_counts.loc[0])),
        ("Fraud duplicates removed", int(original_counts.loc[1] - cleaned_counts.loc[1])),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def save_figure(path: str | Path, dpi: int = 180) -> None:
    """Save the current Matplotlib figure with consistent readable formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, dpi=dpi, bbox_inches="tight")


def format_metric_table(
    dataframe: pd.DataFrame,
    metric_columns: Iterable[str] | None = None,
    decimals: int = 4,
) -> Styler:
    """Return a display-only style without changing full-precision calculations."""
    columns = list(metric_columns or dataframe.select_dtypes(include="number").columns)
    formats = {column: f"{{:.{decimals}f}}" for column in columns}
    return dataframe.style.format(formats)
