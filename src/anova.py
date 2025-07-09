import os

import numpy as np  

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd

from dataloader import load_text_complexity_multi_level_dataset
from datasets import load_dataset


def compute_anova_for_text_complexity():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_complexity_multi_level_dataset()
    )

    train_multiazter_df = pd.read_csv(
        "./datasets/text_complexity_train_multiazter_indicators.csv", index_col="index"
    )
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/text_complexity_train_pucp_indicators.csv", index_col="index"
    )

    train_multiazter_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    repositories = {
        "multiazter": train_multiazter_df,
        "pucp_metrix": train_pucpmetrix_df,
    }

    for repo_name, df_feats in repositories.items():
        vt = VarianceThreshold(threshold=0.0)
        df_feats = pd.DataFrame(
            vt.fit_transform(df_feats), columns=df_feats.columns[vt.get_support()]
        )

        feature_names = df_feats.columns
        selector = SelectKBest(score_func=f_classif, k="all")

        selector.fit(df_feats.to_numpy(), train_labels)

        f_scores = selector.scores_
        p_values = selector.pvalues_

        feature_ranking = sorted(
            zip(feature_names, f_scores, p_values), key=lambda tpl: tpl[1], reverse=True
        )

        print(f"ANOVA for {repo_name} features:")
        for name, f, p in feature_ranking:
            print(f"\t{name:20s}  F-score = {f:.2f}  p-value = {p:.3g}")

        pd.DataFrame(feature_ranking, columns=["feature", "f-score", "p-value"]).to_csv(
            f"./results/anova_text_complexity_{repo_name}_features.csv", index=False
        )


def compute_anova_for_autextification():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )

    train_multiazter_df = pd.read_csv(
        "./datasets/autextification_train_multiazter_indicators.csv", index_col="index"
    )
    train_cohmetrix_df = pd.read_csv(
        "./datasets/autextification_train_cohmetrix_indicators.csv", index_col="index"
    )
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_train_pucp_indicators.csv", index_col="index"
    )

    train_labels = [data["label"] for data in train_dataset]

    repositories = {
        "multiazter": train_multiazter_df,
        "coh_metrix": train_cohmetrix_df,
        "pucp_metrix": train_pucpmetrix_df,
    }

    for repo_name, df_feats in repositories.items():
        vt = VarianceThreshold(threshold=0.0)
        df_feats = pd.DataFrame(
            vt.fit_transform(df_feats), columns=df_feats.columns[vt.get_support()]
        )

        feature_names = df_feats.columns
        selector = SelectKBest(score_func=f_classif, k="all")

        selector.fit(df_feats.to_numpy(), train_labels)

        f_scores = selector.scores_
        p_values = selector.pvalues_

        feature_ranking = sorted(
            zip(feature_names, f_scores, p_values), key=lambda tpl: tpl[1], reverse=True
        )

        print(f"ANOVA for {repo_name} features:")
        for name, f, p in feature_ranking:
            print(f"\t{name:20s}  F-score = {f:.2f}  p-value = {p:.3g}")

        pd.DataFrame(feature_ranking, columns=["feature", "f-score", "p-value"]).to_csv(
            f"./results/anova_autextification_{repo_name}_features.csv", index=False
        )


if __name__ == "__main__":
    compute_anova_for_autextification()
    # compute_anova_for_text_complexity()
