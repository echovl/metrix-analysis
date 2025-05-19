import pandas as pd


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
