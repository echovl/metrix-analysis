import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def merge_scores(scores: list[dict], labels: list[str]):
    """
    This will merge all scores into a single dictionary, adding the label as a prefix in the corresponding score
    """
    merged_scores = {}
    for idx, label in enumerate(labels):
        score = scores[idx]
        for key, value in score.items():
            merged_scores[f"{label}_{key}"] = value
    return merged_scores


def compute_evaluation_scores(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    human_f1, gen_f1 = f1_score(y_true, y_pred, average=None)
    human_rec, gen_rec = recall_score(y_true, y_pred, average=None)
    human_prec, gen_prec = precision_score(y_true, y_pred, average=None)
    return {
        "f1_macro": f1_macro,
        "human_f1": human_f1,
        "gen_f1": gen_f1,
        "human_precision": human_prec,
        "gen_precision": gen_prec,
        "human_recall": human_rec,
        "gen_recall": gen_rec,
    }


def get_cohmetrix_dataset_grouped():
    train_cohmetrix_df = pd.read_csv(
        "./data/train_coh_metrix_metrics.csv", index_col="index"
    )
    test_cohmetrix_df = pd.read_csv(
        "./data/test_coh_metrix_metrics.csv", index_col="index"
    )

    cohmetrix_groups_df = pd.read_csv("./data/cohmetrix_groups.csv")
    cohmetrix_groups_df = (
        cohmetrix_groups_df.groupby("group")["metric"].apply(list).reset_index()
    )

    datasets = {}
    for group_name in cohmetrix_groups_df["group"].unique():
        metrics = cohmetrix_groups_df[cohmetrix_groups_df["group"] == group_name][
            "metric"
        ].tolist()[0]

        train_df = train_cohmetrix_df[metrics]
        test_df = test_cohmetrix_df[metrics]

        train_features = train_df.to_numpy()
        test_features = test_df.to_numpy()

        datasets[group_name.lower()] = {
            "train_features": train_features,
            "test_features": test_features,
        }

    return datasets


def get_multiazter_dataset_grouped():
    train_multiazter_df = pd.read_csv(
        "./data/train_multiazter_metrics.csv", index_col="index"
    )
    test_multiazter_df = pd.read_csv(
        "./data/test_multiazter_metrics.csv", index_col="index"
    )

    multiazter_groups_df = pd.read_csv("./data/multiazter_groups.csv")
    multiazter_groups_df = (
        multiazter_groups_df.groupby("group")["metric"].apply(list).reset_index()
    )

    datasets = {}
    for group_name in multiazter_groups_df["group"].unique():
        metrics = multiazter_groups_df[multiazter_groups_df["group"] == group_name][
            "metric"
        ].tolist()[0]

        train_df = train_multiazter_df[metrics]
        test_df = test_multiazter_df[metrics]

        train_features = train_df.to_numpy()
        test_features = test_df.to_numpy()

        datasets[group_name.lower()] = {
            "train_features": train_features,
            "test_features": test_features,
        }

    return datasets
