import os
from pprint import pprint

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from dataloader import (
    load_autextification_cohmetrix_features,
    load_autextification_dataset,
    load_autextification_multiazter_features,
    load_autextification_pucp_features,
    load_text_complexity_dataset,
    load_text_complexity_multiazter_features,
    load_text_complexity_pucp_features,
)

LABEL_SIMPLE = 1
LABEL_COMPLEX = 0


def train_model(
    repository_name: str,
    train_features: np.ndarray,
    train_labels: [int],
    val_features: np.ndarray,
    val_labels: [int],
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
    training_output.to_csv(f"./results/text_complexity_{repository_name}_ml.csv")

    print(f"Training results for {repository_name}:")
    print(training_output.head())


def train_ml_models():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_complexity_dataset()
    )

    train_multiazter_features, val_multiazter_features, test_multiazter_features = (
        load_text_complexity_multiazter_features()
    )
    train_pucpmetrix_features, val_pucpmetrix_features, test_pucpmetrix_features = (
        load_text_complexity_pucp_features()
    )

    print("Multiazter train features shape:", train_multiazter_features.shape)
    print("Multiazter val features shape:", val_multiazter_features.shape)
    print("Multiazter test features shape:", test_multiazter_features.shape)
    print("PUCPMetrix train features shape:", train_pucpmetrix_features.shape)
    print("PUCPMetrix val features shape:", val_pucpmetrix_features.shape)
    print("PUCPMetrix test features shape:", test_pucpmetrix_features.shape)

    train_model(
        "multiazter",
        train_multiazter_features,
        train_labels,
        val_multiazter_features,
        val_labels,
        test_multiazter_features,
        test_labels,
    )

    train_model(
        "pucp_metrix",
        train_pucpmetrix_features,
        train_labels,
        val_pucpmetrix_features,
        val_labels,
        test_pucpmetrix_features,
        test_labels,
    )


if __name__ == "__main__":
    train_ml_models()
