import pandas as pd

from datasets import load_dataset


def load_autextification_dataset():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_texts = [data["text"] for data in train_dataset]
    test_texts = [data["text"] for data in test_dataset]
    train_labels = [data["label"] for data in train_dataset]
    test_labels = [data["label"] for data in test_dataset]

    return train_texts, train_labels, test_texts, test_labels


def load_autextification_multiazter_features():
    train_multiazter_df = pd.read_csv(
        "./datasets/autextification_train_multiazter_indicators.csv", index_col="index"
    )
    test_multiazter_df = pd.read_csv(
        "./datasets/autextification_test_multiazter_indicators.csv", index_col="index"
    )

    return train_multiazter_df.to_numpy(), test_multiazter_df.to_numpy()


def load_autextification_cohmetrix_features():
    train_cohmetrix_df = pd.read_csv(
        "./datasets/autextification_train_cohmetrix_indicators.csv", index_col="index"
    )
    test_cohmetrix_df = pd.read_csv(
        "./datasets/autextification_test_cohmetrix_indicators.csv", index_col="index"
    )

    return train_cohmetrix_df.to_numpy(), test_cohmetrix_df.to_numpy()


def load_autextification_pucp_features():
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_train_pucp_indicators.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_test_pucp_indicators.csv", index_col="index"
    )

    return train_pucpmetrix_df.to_numpy(), test_pucpmetrix_df.to_numpy()
