import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset


def load_text_complexity_multi_level_dataset():
    cm_dataset = load_dataset("lmvasque/coh-metrix-esp")["train"]
    hc_dataset = load_dataset("lmvasque/hablacultura")["train"]
    kw_dataset = load_dataset("lmvasque/kwiziq")["train"]

    cm_df = pd.DataFrame({"text": cm_dataset["text"], "level": cm_dataset["level"]})
    cm_df["level"] = cm_df["level"].apply(
        lambda lvl: "basic" if lvl == "simple" else "advanced"
    )

    hc_df = pd.DataFrame({"text": hc_dataset["text"], "level": hc_dataset["level-3"]})
    kw_df = pd.DataFrame({"text": kw_dataset["text"], "level": kw_dataset["level-3"]})
    caes_df = pd.read_json("./data/caes.jsonl", lines=True)

    print(set(cm_df["level"]))
    print(set(hc_df["level"]))
    print(set(kw_df["level"]))
    print(set(caes_df["level-3"].tolist()))

    label_map = {"intermediate": 1, "basic": 0, "advanced": 2}

    cm_df["level"] = cm_df["level"].map(label_map)
    hc_df["level"] = hc_df["level"].map(label_map)
    kw_df["level"] = kw_df["level"].map(label_map)
    caes_df["level-3"] = caes_df["level-3"].map(label_map)

    cm_df.dropna(inplace=True)
    hc_df.dropna(inplace=True)
    kw_df.dropna(inplace=True)
    caes_df.dropna(subset=["text", "level-3"], inplace=True)

    # Drop texts with less than 5 characters
    caes_df = caes_df[caes_df["text"].str.len() > 5].reset_index(drop=True)

    cm_texts = [txt for txt in cm_dataset["text"]]
    hc_texts = [txt for txt in hc_dataset["text"]]
    kw_texts = [txt for txt in kw_dataset["text"]]
    caes_texts = caes_df["text"].tolist()

    cm_train_texts, cm_rest_texts, cm_train_labels, cm_rest_labels = train_test_split(
        cm_texts, cm_df["level"].tolist(), test_size=0.20, random_state=42
    )
    cm_test_texts, cm_val_texts, cm_test_labels, cm_val_labels = train_test_split(
        cm_rest_texts, cm_rest_labels, test_size=0.50, random_state=42
    )

    hc_train_texts, hc_rest_texts, hc_train_labels, hc_rest_labels = train_test_split(
        hc_texts, hc_df["level"].tolist(), test_size=0.20, random_state=42
    )
    hc_test_texts, hc_val_texts, hc_test_labels, hc_val_labels = train_test_split(
        hc_rest_texts, hc_rest_labels, test_size=0.50, random_state=42
    )

    kw_train_texts, kw_rest_texts, kw_train_labels, kw_rest_labels = train_test_split(
        kw_texts, kw_df["level"].tolist(), test_size=0.20, random_state=42
    )
    kw_test_texts, kw_val_texts, kw_test_labels, kw_val_labels = train_test_split(
        kw_rest_texts, kw_rest_labels, test_size=0.50, random_state=42
    )

    caes_train_texts, caes_rest_texts, caes_train_labels, caes_rest_labels = (
        train_test_split(
            caes_texts, caes_df["level-3"].tolist(), test_size=0.20, random_state=42
        )
    )
    caes_test_texts, caes_val_texts, caes_test_labels, caes_val_labels = (
        train_test_split(
            caes_rest_texts, caes_rest_labels, test_size=0.50, random_state=42
        )
    )

    print("CM train texts:", len(cm_train_texts))
    print("CM val texts:", len(cm_val_texts))
    print("CM test texts:", len(cm_test_texts))
    print("HC train texts:", len(hc_train_texts))
    print("HC val texts:", len(hc_val_texts))
    print("HC test texts:", len(hc_test_texts))
    print("KW train texts:", len(kw_train_texts))
    print("KW val texts:", len(kw_val_texts))
    print("KW test texts:", len(kw_test_texts))
    print("CAES train texts:", len(caes_train_texts))
    print("CAES val texts:", len(caes_val_texts))
    print("CAES test texts:", len(caes_test_texts))

    merged_train_texts = (
        cm_train_texts + hc_train_texts + kw_train_texts + caes_train_texts
    )
    merged_val_texts = cm_val_texts + hc_val_texts + kw_val_texts + caes_val_texts
    merged_test_texts = cm_test_texts + hc_test_texts + kw_test_texts + caes_test_texts

    merged_train_labels = (
        cm_train_labels + hc_train_labels + kw_train_labels + caes_train_labels
    )
    merged_val_labels = cm_val_labels + hc_val_labels + kw_val_labels + caes_val_labels
    merged_test_labels = (
        cm_test_labels + hc_test_labels + kw_test_labels + caes_test_labels
    )

    pd.DataFrame({"text": merged_train_texts, "label": merged_train_labels}).to_csv(
        "./datasets/text_complexity_multi_level_train.csv", index=False
    )
    pd.DataFrame({"text": merged_val_texts, "label": merged_val_labels}).to_csv(
        "./datasets/text_complexity_multi_level_val.csv", index=False
    )
    pd.DataFrame({"text": merged_test_texts, "label": merged_test_labels}).to_csv(
        "./datasets/text_complexity_multi_level_test.csv", index=False
    )

    return (
        merged_train_texts,
        merged_train_labels,
        merged_val_texts,
        merged_val_labels,
        merged_test_texts,
        merged_test_labels,
    )


def load_text_complexity_dataset():
    cm_dataset = load_dataset("lmvasque/coh-metrix-esp")["train"]
    hc_dataset = load_dataset("lmvasque/hablacultura")["train"]
    kw_dataset = load_dataset("lmvasque/kwiziq")["train"]

    cm_df = pd.DataFrame({"text": cm_dataset["text"], "level": cm_dataset["level"]})
    hc_df = pd.DataFrame({"text": hc_dataset["text"], "level": hc_dataset["level"]})
    kw_df = pd.DataFrame({"text": kw_dataset["text"], "level": kw_dataset["level"]})
    caes_df = pd.read_json("./data/caes.jsonl", lines=True)

    cm_df.dropna(inplace=True)
    hc_df.dropna(inplace=True)
    kw_df.dropna(inplace=True)
    caes_df.dropna(subset=["text", "level"], inplace=True)

    # Drop texts with less than 5 characters
    cm_df = cm_df[cm_df["text"].str.len() > 5].reset_index(drop=True)
    hc_df = hc_df[hc_df["text"].str.len() > 5].reset_index(drop=True)
    kw_df = kw_df[kw_df["text"].str.len() > 5].reset_index(drop=True)
    caes_df = caes_df[caes_df["text"].str.len() > 5].reset_index(drop=True)

    cm_texts = [txt for txt in cm_dataset["text"]]
    hc_texts = [txt for txt in hc_dataset["text"]]
    kw_texts = [txt for txt in kw_dataset["text"]]
    caes_texts = caes_df["text"].tolist()

    cm_train_texts, cm_rest_texts, cm_train_labels, cm_rest_labels = train_test_split(
        cm_texts, cm_df["level"].tolist(), test_size=0.20, random_state=42
    )
    cm_test_texts, cm_val_texts, cm_test_labels, cm_val_labels = train_test_split(
        cm_rest_texts, cm_rest_labels, test_size=0.50, random_state=42
    )

    hc_train_texts, hc_rest_texts, hc_train_labels, hc_rest_labels = train_test_split(
        hc_texts, hc_df["level"].tolist(), test_size=0.20, random_state=42
    )
    hc_test_texts, hc_val_texts, hc_test_labels, hc_val_labels = train_test_split(
        hc_rest_texts, hc_rest_labels, test_size=0.50, random_state=42
    )

    kw_train_texts, kw_rest_texts, kw_train_labels, kw_rest_labels = train_test_split(
        kw_texts, kw_df["level"].tolist(), test_size=0.20, random_state=42
    )
    kw_test_texts, kw_val_texts, kw_test_labels, kw_val_labels = train_test_split(
        kw_rest_texts, kw_rest_labels, test_size=0.50, random_state=42
    )

    caes_train_texts, caes_rest_texts, caes_train_labels, caes_rest_labels = (
        train_test_split(
            caes_texts, caes_df["level"].tolist(), test_size=0.20, random_state=42
        )
    )
    caes_test_texts, caes_val_texts, caes_test_labels, caes_val_labels = (
        train_test_split(
            caes_rest_texts, caes_rest_labels, test_size=0.50, random_state=42
        )
    )

    print("CM train texts:", len(cm_train_texts))
    print("CM val texts:", len(cm_val_texts))
    print("CM test texts:", len(cm_test_texts))
    print("HC train texts:", len(hc_train_texts))
    print("HC val texts:", len(hc_val_texts))
    print("HC test texts:", len(hc_test_texts))
    print("KW train texts:", len(kw_train_texts))
    print("KW val texts:", len(kw_val_texts))
    print("KW test texts:", len(kw_test_texts))
    print("CAES train texts:", len(caes_train_texts))
    print("CAES val texts:", len(caes_val_texts))
    print("CAES test texts:", len(caes_test_texts))

    merged_train_texts = (
        cm_train_texts + hc_train_texts + kw_train_texts + caes_train_texts
    )
    merged_val_texts = cm_val_texts + hc_val_texts + kw_val_texts + caes_val_texts
    merged_test_texts = cm_test_texts + hc_test_texts + kw_test_texts + caes_test_texts

    merged_train_labels = (
        cm_train_labels + hc_train_labels + kw_train_labels + caes_train_labels
    )
    merged_val_labels = cm_val_labels + hc_val_labels + kw_val_labels + caes_val_labels
    merged_test_labels = (
        cm_test_labels + hc_test_labels + kw_test_labels + caes_test_labels
    )

    pd.DataFrame({"text": merged_train_texts, "label": merged_train_labels}).to_csv(
        "./datasets/text_complexity_train.csv", index=False
    )
    pd.DataFrame({"text": merged_val_texts, "label": merged_val_labels}).to_csv(
        "./datasets/text_complexity_val.csv", index=False
    )
    pd.DataFrame({"text": merged_test_texts, "label": merged_test_labels}).to_csv(
        "./datasets/text_complexity_test.csv", index=False
    )

    le = LabelEncoder()
    le.fit(merged_train_labels)

    print("Labels encoding:", le.classes_, le.transform(le.classes_))

    return (
        merged_train_texts,
        le.transform(merged_train_labels),
        merged_val_texts,
        le.transform(merged_val_labels),
        merged_test_texts,
        le.transform(merged_test_labels),
    )


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


def load_text_complexity_pucp_features():
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/text_complexity_train_pucp_indicators.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv(
        "./datasets/text_complexity_test_pucp_indicators.csv", index_col="index"
    )
    val_pucpmetrix_df = pd.read_csv(
        "./datasets/text_complexity_val_pucp_indicators.csv", index_col="index"
    )

    return (
        train_pucpmetrix_df.to_numpy(),
        val_pucpmetrix_df.to_numpy(),
        test_pucpmetrix_df.to_numpy(),
    )


def load_text_complexity_multiazter_features():
    train_multiazter_df = pd.read_csv(
        "./datasets/text_complexity_train_multiazter_indicators.csv", index_col="index"
    )
    test_multiazter_df = pd.read_csv(
        "./datasets/text_complexity_test_multiazter_indicators.csv", index_col="index"
    )
    val_multiazter_df = pd.read_csv(
        "./datasets/text_complexity_val_multiazter_indicators.csv", index_col="index"
    )

    train_multiazter_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    test_multiazter_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)  
    val_multiazter_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    return (
        train_multiazter_df.to_numpy(),
        val_multiazter_df.to_numpy(),
        test_multiazter_df.to_numpy(),
    )


if __name__ == "__main__":
    load_text_complexity_dataset()
