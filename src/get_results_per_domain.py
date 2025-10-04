import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from datasets import load_dataset

test_dataset = load_dataset("symanto/autextification2023", "detection_es", split="test")

test_labels = np.array([data["label"] for data in test_dataset])
test_domains = np.array([data["domain"] for data in test_dataset])

print("Available domains:", set(test_domains))

# Find all autextification prediction files
prediction_files = glob.glob("./results/*_test_predictions.csv")
print(f"\nFound {len(prediction_files)} autextification prediction files:")
for file in prediction_files:
    print(f"  - {file}")

# Initialize results storage
results = {}
unique_domains = np.unique(test_domains)

# Analyze each prediction file
for pred_file in prediction_files:
    # Extract model name from filename
    model_name = (
        os.path.basename(pred_file)
        .replace("autextification_", "")
        .replace("_test_predictions.csv", "")
    )

    print(f"Analyzing {model_name}")

    # Load predictions
    pred_df = pd.read_csv(pred_file)
    predictions = np.array(pred_df["prediction"].tolist())

    # Calculate overall metrics
    overall_precision = precision_score(test_labels, predictions)
    overall_recall = recall_score(test_labels, predictions)
    overall_f1 = f1_score(test_labels, predictions)

    # Calculate metrics per domain
    domain_metrics = {}
    for domain in unique_domains:
        domain_mask = test_domains == domain
        domain_labels = test_labels[domain_mask]
        domain_predictions = predictions[domain_mask]

        domain_precision = precision_score(domain_labels, domain_predictions)
        domain_recall = recall_score(domain_labels, domain_predictions)
        domain_f1 = f1_score(domain_labels, domain_predictions)

        domain_metrics[domain] = {
            "precision": domain_precision,
            "recall": domain_recall,
            "f1": domain_f1,
        }

    results[model_name] = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "domain_metrics": domain_metrics,
    }

    print(f"\n{model_name}:")
    print(
        f"  Overall - Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}, F1: {overall_f1:.3f}"
    )
    for domain in unique_domains:
        metrics = domain_metrics[domain]
        print(
            f"  {domain} - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
        )

# Create visualization showing one metric per chart with bars for each domain
# Layout: 2 charts in first row, 1 chart in second row
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Autextification Models: Performance Metrics by Domain', fontsize=16, fontweight='bold')

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Prepare data for plotting
model_names = list(results.keys())
metrics = ["Precision", "Recall", "F1"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

# Create subplot for each metric
for metric_idx, metric in enumerate(metrics):
    ax = axes_flat[metric_idx]

    # Set up the bar positions
    x = np.arange(len(unique_domains))
    width = 0.8 / len(model_names)

    # Plot bars for each model within each domain
    for i, model in enumerate(model_names):
        metric_key = metric.lower()
        domain_values = [
            results[model]["domain_metrics"][domain][metric_key]
            for domain in unique_domains
        ]

        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, domain_values, width, label=model, alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, domain_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Customize the subplot
    ax.set_xlabel("Domains", fontsize=12)
    ax.set_ylabel(f"{metric} Score", fontsize=12)
    ax.set_title(f"{metric} Performance by Domain", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_domains)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

    # Add overall performance information
    overall_values = [results[model][f"overall_{metric_key}"] for model in model_names]
    best_model_idx = np.argmax(overall_values)
    best_value = overall_values[best_model_idx]
    overall_text = f"Best Overall: {model_names[best_model_idx]} ({best_value:.3f})"

    # Add text box with overall information
    ax.text(
        0.02,
        0.98,
        overall_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

# Hide the unused subplot (4th one)
axes_flat[3].set_visible(False)

plt.tight_layout()
# plt.savefig(
#     "./results/autextification_metrics_by_domain.png", dpi=300, bbox_inches="tight"
# )
plt.show()
