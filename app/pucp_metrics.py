import os
from pprint import pprint

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from collections import OrderedDict

import numpy as np
import pandas as pd
from berta import train_roberta_metrics_model
from berta_grouped import train_berta_extended_model_keras
from iapucp_metrix.analyzer import Analyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from datasets import load_dataset

LABEL_HUMAN = 0
LABEL_GENERATED = 1
SAMPLE_SIZE = 50


pucp_metrix = Analyzer()


def pucp_metrics(texts: [str]) -> list[OrderedDict[str, float]]:
    metrics = pucp_metrix.compute_metrics(texts, workers=16, batch_size=500)

    # replace None values with 0
    for m in metrics:
        for k, v in m.items():
            if v is None:
                m[k] = 0

    return [OrderedDict(m) for m in metrics]


def compute_and_save_pucp_metrics():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )

    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_texts = [data["text"] for data in train_dataset]
    print(train_texts[0])
    test_texts = [data["text"] for data in test_dataset]

    print("Number of Train texts:", len(train_texts))
    print("Number of Test texts:", len(test_texts))

    train_pucp_metrics = pucp_metrics(train_texts)
    test_pucp_metrics = pucp_metrics(test_texts)

    train_pucp_df = pd.DataFrame(train_pucp_metrics)
    train_pucp_df.to_csv("train_pucp_metrics.csv", index_label="index")

    test_pucp_df = pd.DataFrame(test_pucp_metrics)
    test_pucp_df.to_csv("test_pucp_metrics.csv", index_label="index")


def train_model(
    repository_name: str,
    train_features: np.ndarray,
    train_labels: [int],
    test_features: np.ndarray,
    test_labels: [int],
):
    xgb_pipeline = Pipeline([("clf", XGBClassifier())])
    xgb_parameters = {
        "clf__max_depth": range(1, 8, 2),
        "clf__n_estimators": range(20, 200, 40),
        "clf__learning_rate": [0.1, 0.01, 0.05],
    }

    svc_pipeline = Pipeline([("scaler", RobustScaler()), ("clf", LinearSVC())])
    svc_parameters = {
        "clf__C": range(1, 8, 1),
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [False],
        "clf__max_iter": [40000],
    }

    lr_pipeline = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression())])
    lr_parameters = {
        "clf__C": range(1, 24, 3),
        "clf__dual": [False],
        "clf__max_iter": [20000],
    }
    rf_pipeline = Pipeline([("clf", RandomForestClassifier())])
    rf_parameters = {
        "clf__n_estimators": range(20, 200, 40),
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_features": ["sqrt", "log2"],
        "clf__max_depth": range(1, 8, 2),
    }

    xgb_model = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_parameters,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    svc_model = GridSearchCV(
        estimator=svc_pipeline,
        param_grid=svc_parameters,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    lr_model = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=lr_parameters,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    rf_model = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_parameters,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    models = [
        ("lr", lr_model),
        ("xgb", xgb_model),
        ("svm", svc_model),
        ("rf", rf_model),
    ]

    print(f"Training models with {repository_name}...")

    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.20, random_state=42
    )

    def get_scores(y_true, y_pred):
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

    model_results = {
        "model": [model_name for model_name, _ in models],
        "cv_f1_macro": [],
    }

    for model_name, model in models:
        print(f"Training {model_name} model")

        X, y = shuffle(train_features, train_labels, random_state=42)

        model.fit(X, y)

        # joblib.dump(model, f"./models/{repository_name}_{model_name}.pkl", compress=1)

        train_predicted = model.predict(train_features)
        test_predicted = model.predict(test_features)
        val_predicted = model.predict(val_features)

        domains = {
            "train": [train_labels, train_predicted],
            "test": [test_labels, test_predicted],
            "val": [val_labels, val_predicted],
        }

        for domain, data in domains.items():
            y_true, y_pred = data
            scores = get_scores(y_true, y_pred)

            print(f"{model_name} {domain} scores: \n")
            pprint(scores)
            print("\n")

            for key, value in scores.items():
                result_key = f"{domain}_{key}"

                if result_key not in model_results:
                    model_results[result_key] = []
                model_results[result_key].append(value)

        model_results["cv_f1_macro"].append(model.best_score_)

    training_output = pd.DataFrame(model_results)
    training_output.to_csv(f"./results/autextification_{repository_name}_ml.csv")

    print(f"Training results for {repository_name}:")
    print(training_output.head())


def compute_anova_for_all_metrics():
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
            f"./results/anova_{repo_name}_features.csv", index=False
        )


def train_ml_models():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_multiazter_df = pd.read_csv(
        "./datasets/autextification_train_multiazter_indicators.csv", index_col="index"
    )
    test_multiazter_df = pd.read_csv(
        "./datasets/autextification_test_multiazter_indicators.csv", index_col="index"
    )
    train_cohmetrix_df = pd.read_csv(
        "./datasets/autextification_train_cohmetrix_indicators.csv", index_col="index"
    )
    test_cohmetrix_df = pd.read_csv(
        "./datasets/autextification_test_cohmetrix_indicators.csv", index_col="index"
    )
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_train_pucp_indicators.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_test_pucp_indicators.csv", index_col="index"
    )

    train_multiazter_ratios = pd.read_csv(
        "./data/train_multiazter_metrics.csv",
        index_col="index",
    )
    print(
        "Train multiazter only ratios shape:", train_multiazter_ratios.to_numpy().shape
    )

    columns_diff = set(train_multiazter_df.columns).symmetric_difference(
        set(train_multiazter_ratios.columns)
    )
    print("Columns diff:", columns_diff)

    train_multiazter_features = train_multiazter_df.to_numpy()
    test_multiazter_features = test_multiazter_df.to_numpy()
    train_cohmetrix_features = train_cohmetrix_df.to_numpy()
    test_cohmetrix_features = test_cohmetrix_df.to_numpy()
    train_pucpmetrix_features = train_pucpmetrix_df.to_numpy()
    test_pucpmetrix_features = test_pucpmetrix_df.to_numpy()

    print("Multiazter features shape:", train_multiazter_features.shape)
    print("Multiazter test features shape:", test_multiazter_features.shape)
    print("Cohmetrix train features shape:", train_cohmetrix_features.shape)
    print("Cohmetrix test features shape:", test_cohmetrix_features.shape)
    print("PUCPMetrix features shape:", train_pucpmetrix_features.shape)
    print("PUCPMetrix test features shape:", test_pucpmetrix_features.shape)

    train_labels = [data["label"] for data in train_dataset]
    test_labels = [data["label"] for data in test_dataset]

    train_model(
        "multiazter",
        train_multiazter_features,
        train_labels,
        test_multiazter_features,
        test_labels,
    )

    train_model(
        "coh_metrix",
        train_cohmetrix_features,
        train_labels,
        test_cohmetrix_features,
        test_labels,
    )
    train_model(
        "pucp_metrix",
        train_pucpmetrix_features,
        train_labels,
        test_pucpmetrix_features,
        test_labels,
    )


def train_berta_pucp_model():
    train_pucpmetrix_df = pd.read_csv(
        "./data/train_pucp_metrics.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv("./data/test_pucp_metrics.csv", index_col="index")

    print("Train pucp metrics shape:", train_pucpmetrix_df.to_numpy().shape)
    print("Test pucp metrics shape:", test_pucpmetrix_df.to_numpy().shape)

    train_roberta_metrics_model(
        train_pucpmetrix_df.to_numpy(),
        test_pucpmetrix_df.to_numpy(),
    )


def train_multiazter_model():
    train_pucpmetrix_df = pd.read_csv(
        "./data/train_multiazter_metrics.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv(
        "./data/test_multiazter_metrics.csv", index_col="index"
    )

    train_berta_extended_model_keras(
        "multiazter",
        train_pucpmetrix_df.to_numpy(),
        test_pucpmetrix_df.to_numpy(),
    )


if __name__ == "__main__":
    # train_berta_multiazter_model()
    # compute_and_save_pucp_metrics()
    # train_ml_models()
    compute_anova_for_all_metrics()
    # train_berta_pucp_model()
    # train_multiazter_model()
