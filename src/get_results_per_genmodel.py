import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from datasets import load_dataset

test_dataset = load_dataset("symanto/autextification2023", "detection_es", split="test")

test_labels = np.array([data["label"] for data in test_dataset])
test_gen_models = np.array([data["model"] for data in test_dataset])

print("Available gen_models:", set(test_gen_models))

# Find all autextification prediction files
prediction_files = glob.glob("./results/autextification_*_test_predictions.csv")
print(f"\nFound {len(prediction_files)} autextification prediction files:")
for file in prediction_files:
    print(f"  - {file}")

# Initialize results storage
results = {}
unique_gen_models = np.unique(test_gen_models)

# Analyze each prediction file
for pred_file in prediction_files:
    # Extract model name from filename
    model_name = (
        os.path.basename(pred_file)
        .replace("autextification_", "")
        .replace("_test_predictions.csv", "")
    )

    # Load predictions
    pred_df = pd.read_csv(pred_file)
    predictions = np.array(pred_df["prediction"].tolist())

    # Calculate overall metrics
    overall_precision = precision_score(test_labels, predictions)
    overall_recall = recall_score(test_labels, predictions)
    overall_f1 = f1_score(test_labels, predictions, average="macro")

    # Calculate metrics per gen_model
    gen_model_metrics = {}
    for gen_model in unique_gen_models:
        gen_model_mask = test_gen_models == gen_model
        gen_model_labels = test_labels[gen_model_mask]
        gen_model_predictions = predictions[gen_model_mask]

        gen_model_precision = precision_score(gen_model_labels, gen_model_predictions)
        gen_model_recall = recall_score(gen_model_labels, gen_model_predictions)
        gen_model_f1 = f1_score(gen_model_labels, gen_model_predictions)

        gen_model_metrics[gen_model] = {
            "precision": gen_model_precision,
            "recall": gen_model_recall,
            "f1": gen_model_f1,
        }

    results[model_name] = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "gen_model_metrics": gen_model_metrics,
    }

    print(f"\n{model_name}:")
    print(
        f"  Overall - Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}, F1: {overall_f1:.3f}"
    )
    for gen_model in unique_gen_models:
        metrics = gen_model_metrics[gen_model]
        print(
            f"  {gen_model} - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
        )

# Create visualization showing one metric per chart with bars for each gen_model
# Layout: 2 charts in first row, 1 chart in second row
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
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
    x = np.arange(len(unique_gen_models))
    width = 0.8 / len(model_names)

    # Plot bars for each model within each gen_model
    for i, model in enumerate(model_names):
        metric_key = metric.lower()
        gen_model_values = [
            results[model]["gen_model_metrics"][gen_model][metric_key]
            for gen_model in unique_gen_models
        ]

        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, gen_model_values, width, label=model, alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, gen_model_values):
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
    ax.set_xlabel("Gen Models", fontsize=12)
    ax.set_ylabel(f"{metric} Score", fontsize=12)
    ax.set_title(f"{metric} Performance by Gen Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_gen_models)
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
# axes_flat[3].set_visible(False)

plt.tight_layout()
plt.savefig(
    "./results/autextification_metrics_by_gen_model.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
summary_data = []
for model_name in model_names:
    row = {
        "Model": model_name,
        "Overall_Precision": results[model_name]["overall_precision"],
        "Overall_Recall": results[model_name]["overall_recall"],
        "Overall_F1": results[model_name]["overall_f1"],
    }
    for gen_model in unique_gen_models:
        gen_model_metrics = results[model_name]["gen_model_metrics"][gen_model]
        row[f"{gen_model}_Precision"] = gen_model_metrics["precision"]
        row[f"{gen_model}_Recall"] = gen_model_metrics["recall"]
        row[f"{gen_model}_F1"] = gen_model_metrics["f1"]
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("Overall_F1", ascending=False)
print(summary_df.to_string(index=False, float_format="%.3f"))

# Save summary to CSV
summary_df.to_csv("./results/autextification_metrics_summary.csv", index=False)
print(f"\nVisualization saved to: ./results/autextification_metrics_by_gen_model.png")
print(f"Summary saved to: ./results/autextification_metrics_summary.csv")
