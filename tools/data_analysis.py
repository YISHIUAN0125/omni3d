import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET_FILES = {
    "SUNRGBD_train": "datasets/Omni3D/SUNRGBD_train.json",
    "SUNRGBD_val": "datasets/Omni3D/SUNRGBD_val.json",
    "SUNRGBD_test": "datasets/Omni3D/SUNRGBD_test.json",

    "Hypersim_train": "datasets/Omni3D/Hypersim_train.json",
    "Hypersim_val": "datasets/Omni3D/Hypersim_val.json",
    "Hypersim_test": "datasets/Omni3D/Hypersim_test.json",

    "ARKitScenes_train": "datasets/Omni3D/ARKitScenes_train.json",
    "ARKitScenes_val": "datasets/Omni3D/ARKitScenes_val.json",
    "ARKitScenes_test": "datasets/Omni3D/ARKitScenes_test.json",
}

DATASET_PREFIXES = [
    "SUNRGBD",
    "Hypersim",
    "ARKitScenes",
]

OUTPUT_DIR = Path("datasets/Omni3D/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_CSV_PATH = OUTPUT_DIR / "visibility_split_statistics.csv"
DATASET_CSV_PATH = OUTPUT_DIR / "visibility_dataset_statistics.csv"
OVERALL_CSV_PATH = OUTPUT_DIR / "visibility_overall_statistics.csv"
SUMMARY_TXT_PATH = OUTPUT_DIR / "visibility_analysis_summary.txt"
FIGURE_PATH = OUTPUT_DIR / "visibility_analysis_merged.png"

MIN_CORRELATION_SAMPLES = 20

# Visibility protocol
HARD_THRESHOLD = 0.3
EASY_THRESHOLD = 0.7


# ============================================================
# Utility functions
# ============================================================

def safe_float(value):
    """
    Convert a value to a finite float.

    Returns
    -------
    float or None
        None is returned when conversion fails or value is NaN/Inf.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def get_dataset_prefix(split_name):
    """
    Convert names such as SUNRGBD_train to SUNRGBD.
    """
    for prefix in DATASET_PREFIXES:
        if split_name.startswith(prefix):
            return prefix

    return split_name.rsplit("_", 1)[0]


def safe_correlation(depths, visibilities):
    """
    Calculate Pearson correlation safely.

    Correlation is undefined when:
    1. There are too few samples.
    2. Depth has zero variance.
    3. Visibility has zero variance.
    """
    depths = np.asarray(depths, dtype=np.float64)
    visibilities = np.asarray(visibilities, dtype=np.float64)

    mask = (
        np.isfinite(depths)
        & np.isfinite(visibilities)
        & (depths > 0)
        & (visibilities >= 0)
        & (visibilities <= 1)
    )

    depths = depths[mask]
    visibilities = visibilities[mask]

    if len(depths) < MIN_CORRELATION_SAMPLES:
        return np.nan, len(depths)

    if np.std(depths) == 0 or np.std(visibilities) == 0:
        return np.nan, len(depths)

    corr = np.corrcoef(depths, visibilities)[0, 1]

    if not np.isfinite(corr):
        return np.nan, len(depths)

    return float(corr), len(depths)


def calculate_visibility_statistics(
    dataset_name,
    total_annotations,
    visibility_values,
    depth_values,
    depth_visibility_values,
    key_present_count,
    missing_key_count,
    unavailable_count,
    outlier_count,
):
    """
    Calculate visibility statistics using legal visibility values only.

    Legal visibility range:
        0 <= visibility <= 1
    """
    vis = np.asarray(visibility_values, dtype=np.float64)

    valid_count = len(vis)

    if total_annotations > 0:
        available_ratio = (
            key_present_count - unavailable_count
        ) / total_annotations

        valid_ratio = valid_count / total_annotations
    else:
        available_ratio = np.nan
        valid_ratio = np.nan

    if valid_count == 0:
        return {
            "dataset": dataset_name,
            "total": total_annotations,
            "visibility_key_present": key_present_count,
            "missing_key": missing_key_count,
            "unavailable_minus1": unavailable_count,
            "outlier_count": outlier_count,
            "available_non_minus1": (
                key_present_count - unavailable_count
            ),
            "available_ratio": available_ratio,
            "valid": 0,
            "valid_ratio": valid_ratio,
            "invalid_ratio": (
                outlier_count / total_annotations
                if total_annotations > 0
                else np.nan
            ),
            "mean_vis": np.nan,
            "median_vis": np.nan,
            "std_vis": np.nan,
            "min_vis": np.nan,
            "max_vis": np.nan,
            "corr": np.nan,
            "corr_samples": 0,
            "extreme": np.nan,
            "heavy": np.nan,
            "moderate": np.nan,
            "low": np.nan,
            "visible": np.nan,
            "full": np.nan,
            "hard_v2": np.nan,
            "moderate_v2": np.nan,
            "easy_v2": np.nan,
        }

    extreme = np.mean(vis < 0.1)

    heavy = np.mean(
        (vis >= 0.1)
        & (vis < 0.3)
    )

    moderate = np.mean(
        (vis >= 0.3)
        & (vis < 0.5)
    )

    low = np.mean(
        (vis >= 0.5)
        & (vis < 0.7)
    )

    visible = np.mean(
        (vis >= 0.7)
        & (vis < 0.9)
    )

    full = np.mean(vis >= 0.9)

    hard_v2 = np.mean(vis < HARD_THRESHOLD)

    moderate_v2 = np.mean(
        (vis >= HARD_THRESHOLD)
        & (vis < EASY_THRESHOLD)
    )

    easy_v2 = np.mean(vis >= EASY_THRESHOLD)

    corr, corr_samples = safe_correlation(
        depth_values,
        depth_visibility_values,
    )

    return {
        "dataset": dataset_name,
        "total": total_annotations,
        "visibility_key_present": key_present_count,
        "missing_key": missing_key_count,
        "unavailable_minus1": unavailable_count,
        "outlier_count": outlier_count,
        "available_non_minus1": (
            key_present_count - unavailable_count
        ),
        "available_ratio": available_ratio,
        "valid": valid_count,
        "valid_ratio": valid_ratio,
        "invalid_ratio": (
            outlier_count / total_annotations
            if total_annotations > 0
            else np.nan
        ),
        "mean_vis": float(np.mean(vis)),
        "median_vis": float(np.median(vis)),
        "std_vis": float(np.std(vis)),
        "min_vis": float(np.min(vis)),
        "max_vis": float(np.max(vis)),
        "corr": corr,
        "corr_samples": corr_samples,
        "extreme": float(extreme),
        "heavy": float(heavy),
        "moderate": float(moderate),
        "low": float(low),
        "visible": float(visible),
        "full": float(full),
        "hard_v2": float(hard_v2),
        "moderate_v2": float(moderate_v2),
        "easy_v2": float(easy_v2),
    }


def add_bar_labels(axis, values, percentage=False):
    """
    Add labels above bars while safely skipping NaN values.
    """
    for index, value in enumerate(values):
        if pd.isna(value):
            continue

        label = f"{value:.1%}" if percentage else f"{value:.3f}"

        axis.text(
            index,
            value + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=18,
        )


# ============================================================
# Raw data containers
# ============================================================

split_results = []

# Preserve legal visibility/depth values for each dataset.
# This allows train/val/test to be pooled before statistics are
# recalculated.
dataset_storage = {
    prefix: {
        "total": 0,
        "visibility_key_present": 0,
        "missing_key": 0,
        "unavailable_minus1": 0,
        "outlier_count": 0,
        "visibility_values": [],
        "depth_values": [],
        "depth_visibility_values": [],
    }
    for prefix in DATASET_PREFIXES
}

existing_file_count = 0


# ============================================================
# Parse each split
# ============================================================

for split_name, json_path_string in DATASET_FILES.items():

    json_path = Path(json_path_string)

    print("=" * 70)
    print(f"Processing: {split_name}")
    print(f"Path: {json_path}")

    if not json_path.exists():
        print("Status: file not found, skipped")
        continue

    existing_file_count += 1

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Status: failed to load JSON: {error}")
        continue

    annotations = data.get("annotations", [])

    if not isinstance(annotations, list):
        print("Status: 'annotations' is not a list, skipped")
        continue

    total = len(annotations)

    visibility_values = []
    depth_values = []
    depth_visibility_values = []

    key_present_count = 0
    missing_key_count = 0
    unavailable_count = 0
    outlier_count = 0

    dataset_prefix = get_dataset_prefix(split_name)

    if dataset_prefix not in dataset_storage:
        dataset_storage[dataset_prefix] = {
            "total": 0,
            "visibility_key_present": 0,
            "missing_key": 0,
            "unavailable_minus1": 0,
            "outlier_count": 0,
            "visibility_values": [],
            "depth_values": [],
            "depth_visibility_values": [],
        }

    for annotation in annotations:

        if not isinstance(annotation, dict):
            missing_key_count += 1
            continue

        if "visibility" not in annotation:
            missing_key_count += 1
            continue

        key_present_count += 1

        visibility = safe_float(annotation.get("visibility"))

        # Invalid nonnumeric, NaN, or Inf visibility
        if visibility is None:
            outlier_count += 1
            continue

        # Omni3D unavailable marker
        if visibility == -1:
            unavailable_count += 1
            continue

        # Only [0, 1] is statistically valid
        if visibility < 0 or visibility > 1:
            outlier_count += 1
            continue

        visibility_values.append(visibility)

        center = annotation.get("center_cam")

        if not isinstance(center, (list, tuple)):
            continue

        if len(center) < 3:
            continue

        depth = safe_float(center[2])

        if depth is None:
            continue

        # Only positive camera depth is valid
        if depth <= 0:
            continue

        depth_values.append(depth)
        depth_visibility_values.append(visibility)

    split_result = calculate_visibility_statistics(
        dataset_name=split_name,
        total_annotations=total,
        visibility_values=visibility_values,
        depth_values=depth_values,
        depth_visibility_values=depth_visibility_values,
        key_present_count=key_present_count,
        missing_key_count=missing_key_count,
        unavailable_count=unavailable_count,
        outlier_count=outlier_count,
    )

    split_results.append(split_result)

    # Pool raw values at dataset level
    storage = dataset_storage[dataset_prefix]

    storage["total"] += total
    storage["visibility_key_present"] += key_present_count
    storage["missing_key"] += missing_key_count
    storage["unavailable_minus1"] += unavailable_count
    storage["outlier_count"] += outlier_count
    storage["visibility_values"].extend(visibility_values)
    storage["depth_values"].extend(depth_values)
    storage["depth_visibility_values"].extend(
        depth_visibility_values
    )

    print(f"Total annotations: {total:,}")
    print(f"Valid visibility: {len(visibility_values):,}")
    print(f"Visibility = -1: {unavailable_count:,}")
    print(f"Outliers: {outlier_count:,}")
    print(f"Valid depth pairs: {len(depth_values):,}")


# ============================================================
# Ensure at least one file was processed
# ============================================================

if existing_file_count == 0:
    raise FileNotFoundError(
        "No dataset JSON file was found. "
        "Please check the paths in DATASET_FILES."
    )

if len(split_results) == 0:
    raise RuntimeError(
        "Dataset files were found, but no valid JSON annotation "
        "file could be processed."
    )


# ============================================================
# Split-level DataFrame
# ============================================================

split_df = pd.DataFrame(split_results)

split_df.to_csv(
    SPLIT_CSV_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Merge train/val/test at dataset level
# ============================================================

dataset_results = []

for dataset_prefix, storage in dataset_storage.items():

    if storage["total"] == 0:
        continue

    dataset_result = calculate_visibility_statistics(
        dataset_name=f"{dataset_prefix}_ALL",
        total_annotations=storage["total"],
        visibility_values=storage["visibility_values"],
        depth_values=storage["depth_values"],
        depth_visibility_values=storage[
            "depth_visibility_values"
        ],
        key_present_count=storage[
            "visibility_key_present"
        ],
        missing_key_count=storage["missing_key"],
        unavailable_count=storage["unavailable_minus1"],
        outlier_count=storage["outlier_count"],
    )

    dataset_results.append(dataset_result)

dataset_df = pd.DataFrame(dataset_results)

dataset_df.to_csv(
    DATASET_CSV_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Overall object-level micro statistics
# ============================================================

overall_total = 0
overall_key_present = 0
overall_missing_key = 0
overall_unavailable = 0
overall_outlier = 0

overall_visibility_values = []
overall_depth_values = []
overall_depth_visibility_values = []

for storage in dataset_storage.values():

    overall_total += storage["total"]
    overall_key_present += storage[
        "visibility_key_present"
    ]
    overall_missing_key += storage["missing_key"]
    overall_unavailable += storage[
        "unavailable_minus1"
    ]
    overall_outlier += storage["outlier_count"]

    overall_visibility_values.extend(
        storage["visibility_values"]
    )

    overall_depth_values.extend(
        storage["depth_values"]
    )

    overall_depth_visibility_values.extend(
        storage["depth_visibility_values"]
    )

overall_result = calculate_visibility_statistics(
    dataset_name="Omni3D_Indoor_Micro",
    total_annotations=overall_total,
    visibility_values=overall_visibility_values,
    depth_values=overall_depth_values,
    depth_visibility_values=overall_depth_visibility_values,
    key_present_count=overall_key_present,
    missing_key_count=overall_missing_key,
    unavailable_count=overall_unavailable,
    outlier_count=overall_outlier,
)

overall_df = pd.DataFrame([overall_result])

overall_df.to_csv(
    OVERALL_CSV_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Dataset-balanced macro statistics
# ============================================================

valid_dataset_df = dataset_df[
    dataset_df["valid"] > 0
].copy()

if len(valid_dataset_df) > 0:
    macro_mean_visibility = valid_dataset_df[
        "mean_vis"
    ].mean()

    macro_hard = valid_dataset_df[
        "hard_v2"
    ].mean()

    macro_moderate = valid_dataset_df[
        "moderate_v2"
    ].mean()

    macro_easy = valid_dataset_df[
        "easy_v2"
    ].mean()
else:
    macro_mean_visibility = np.nan
    macro_hard = np.nan
    macro_moderate = np.nan
    macro_easy = np.nan


# ============================================================
# Print statistics
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.float_format", "{:.6f}".format)

print("\n")
print("=" * 100)
print("SPLIT-LEVEL STATISTICS")
print("=" * 100)
print(split_df)

print("\n")
print("=" * 100)
print("DATASET-LEVEL STATISTICS")
print("=" * 100)
print(dataset_df)

print("\n")
print("=" * 100)
print("OVERALL OBJECT-LEVEL MICRO STATISTICS")
print("=" * 100)
print(overall_df)


# ============================================================
# Prepare paper-oriented plot data
# ============================================================

plot_df = dataset_df.copy()
plot_labels = (
    plot_df["dataset"].str.replace("_ALL", "", regex=False).tolist()
)
x = np.arange(len(plot_df))

# Fraction of all annotations with a usable visibility value in [0, 1].
valid_visibility_values = plot_df["valid_ratio"].to_numpy(dtype=float)
mean_visibility_values = plot_df["mean_vis"].to_numpy(dtype=float)
correlation_values = plot_df["corr"].to_numpy(dtype=float)

hard_values = np.nan_to_num(plot_df["hard_v2"].to_numpy(dtype=float), nan=0.0)
moderate_values = np.nan_to_num(plot_df["moderate_v2"].to_numpy(dtype=float), nan=0.0)
easy_values = np.nan_to_num(plot_df["easy_v2"].to_numpy(dtype=float), nan=0.0)

# ============================================================
# Paper-oriented figure
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Visibility Annotation and Depth Analysis in Omni3D Indoor Datasets",
    fontsize=26,
)

# 1. Availability of usable visibility annotations
axes[0, 0].bar(x, valid_visibility_values)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[0, 0].set_ylim(0, 1.08)
axes[0, 0].set_ylabel("Ratio")
axes[0, 0].set_title("Usable Visibility Annotation Ratio")
for index, value in enumerate(valid_visibility_values):
    if np.isfinite(value):
        axes[0, 0].text(index, value + 0.015, f"{value:.1%}", ha="center", va="bottom", fontsize=13)

# 2. Visibility severity distribution
axes[0, 1].bar(x, hard_values, label="Hard (<0.3)")
axes[0, 1].bar(x, moderate_values, bottom=hard_values, label="Moderate (0.3–0.7)")
axes[0, 1].bar(x, easy_values, bottom=hard_values + moderate_values, label="Easy (≥0.7)")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[0, 1].set_ylim(0, 1.0)
axes[0, 1].set_ylabel("Ratio")
axes[0, 1].set_title("Visibility Severity Distribution")
axes[0, 1].legend(loc="upper right", fontsize=10)
for index in range(len(plot_df)):
    if hard_values[index] >= 0.07:
        axes[0, 1].text(index, hard_values[index] / 2, f"{hard_values[index]:.1%}", ha="center", va="center", fontsize=13)
    if moderate_values[index] >= 0.07:
        axes[0, 1].text(index, hard_values[index] + moderate_values[index] / 2, f"{moderate_values[index]:.1%}", ha="center", va="center", fontsize=13)
    if easy_values[index] >= 0.07:
        axes[0, 1].text(index, hard_values[index] + moderate_values[index] + easy_values[index] / 2, f"{easy_values[index]:.1%}", ha="center", va="center", fontsize=13)

# 3. Mean visibility
axes[1, 0].bar(x, mean_visibility_values)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[1, 0].set_ylim(0, 1.0)
axes[1, 0].set_ylabel("Mean visibility")
axes[1, 0].set_title("Mean Visibility of Usable Annotations")
for index, value in enumerate(mean_visibility_values):
    if np.isfinite(value):
        axes[1, 0].text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=13)

# 4. Depth–visibility relationship
axes[1, 1].bar(x, np.nan_to_num(correlation_values, nan=0.0))
axes[1, 1].axhline(0, linewidth=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(plot_labels, rotation=20, ha="right")
axes[1, 1].set_ylim(-1.0, 1.0)
axes[1, 1].set_ylabel("Pearson correlation (r)")
axes[1, 1].set_title("Center Depth–Visibility Correlation")
for index, value in enumerate(correlation_values):
    if np.isfinite(value):
        y_offset = 0.04 if value >= 0 else -0.07
        axes[1, 1].text(index, value + y_offset, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=13)
    else:
        axes[1, 1].text(index, 0.03, "N/A", ha="center", va="bottom", fontsize=13)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Save text summary
# ============================================================

text_lines = []

text_lines.append("=" * 90)
text_lines.append("OMNI3D INDOOR VISIBILITY ANALYSIS")
text_lines.append("=" * 90)
text_lines.append("")

text_lines.append(
    "Usable visibility rule: 0 <= visibility <= 1"
)
text_lines.append(
    "Depth validity rule: finite center_cam[2] > 0"
)
text_lines.append(
    f"Hard: visibility < {HARD_THRESHOLD}"
)
text_lines.append(
    f"Moderate: {HARD_THRESHOLD} <= visibility "
    f"< {EASY_THRESHOLD}"
)
text_lines.append(
    f"Easy: visibility >= {EASY_THRESHOLD}"
)
text_lines.append("")

text_lines.append("-" * 90)
text_lines.append("DATASET-LEVEL RESULTS")
text_lines.append("-" * 90)

for _, row in dataset_df.iterrows():

    text_lines.append("")
    text_lines.append(str(row["dataset"]))
    text_lines.append(
        f"  Total annotations : {int(row['total']):,}"
    )
    text_lines.append(
        f"  Valid visibility  : {int(row['valid']):,}"
    )
    text_lines.append(
        f"  Valid ratio       : {row['valid_ratio']:.2%}"
    )
    text_lines.append(
        f"  Visibility = -1   : "
        f"{int(row['unavailable_minus1']):,}"
    )
    text_lines.append(
        f"  Outliers          : "
        f"{int(row['outlier_count']):,}"
    )

    if row["valid"] > 0:
        text_lines.append(
            f"  Mean visibility   : {row['mean_vis']:.4f}"
        )
        text_lines.append(
            f"  Median visibility : {row['median_vis']:.4f}"
        )
        text_lines.append(
            f"  Hard              : {row['hard_v2']:.2%}"
        )
        text_lines.append(
            f"  Moderate          : "
            f"{row['moderate_v2']:.2%}"
        )
        text_lines.append(
            f"  Easy              : {row['easy_v2']:.2%}"
        )

        if pd.isna(row["corr"]):
            text_lines.append(
                "  Depth-vis corr    : N/A"
            )
        else:
            text_lines.append(
                f"  Depth-vis corr    : {row['corr']:.4f}"
            )

text_lines.append("")
text_lines.append("-" * 90)
text_lines.append("OVERALL OBJECT-LEVEL MICRO RESULTS")
text_lines.append("-" * 90)
text_lines.append(
    f"Total annotations : {overall_result['total']:,}"
)
text_lines.append(
    f"Valid visibility  : {overall_result['valid']:,}"
)
text_lines.append(
    f"Valid ratio       : {overall_result['valid_ratio']:.2%}"
)
text_lines.append(
    f"Mean visibility   : {overall_result['mean_vis']:.4f}"
)
text_lines.append(
    f"Median visibility : {overall_result['median_vis']:.4f}"
)
text_lines.append(
    f"Hard              : {overall_result['hard_v2']:.2%}"
)
text_lines.append(
    f"Moderate          : "
    f"{overall_result['moderate_v2']:.2%}"
)
text_lines.append(
    f"Easy              : {overall_result['easy_v2']:.2%}"
)
text_lines.append(
    f"Depth-vis corr    : {overall_result['corr']:.4f}"
)

text_lines.append("")
text_lines.append("-" * 90)
text_lines.append("DATASET-BALANCED MACRO RESULTS")
text_lines.append("-" * 90)
text_lines.append(
    f"Mean visibility   : {macro_mean_visibility:.4f}"
)
text_lines.append(
    f"Hard              : {macro_hard:.2%}"
)
text_lines.append(
    f"Moderate          : {macro_moderate:.2%}"
)
text_lines.append(
    f"Easy              : {macro_easy:.2%}"
)

text_lines.append("")
text_lines.append(
    "Note: Micro statistics weight every valid object equally."
)
text_lines.append(
    "Note: Macro statistics weight every dataset equally."
)
text_lines.append(
    "Note: Dataset correlations are recalculated from pooled "
    "raw samples, not averaged across splits."
)

with SUMMARY_TXT_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(text_lines))


# ============================================================
# Final output
# ============================================================

print("\n")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
print(f"Split statistics   : {SPLIT_CSV_PATH}")
print(f"Dataset statistics : {DATASET_CSV_PATH}")
print(f"Overall statistics : {OVERALL_CSV_PATH}")
print(f"Text summary       : {SUMMARY_TXT_PATH}")
print(f"Dashboard figure   : {FIGURE_PATH}")
print("=" * 90)