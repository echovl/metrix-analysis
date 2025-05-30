import os
from pprint import pprint

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFRobertaModel
from xgboost import XGBClassifier

from dataloader import (
    load_autextification_cohmetrix_features,
    load_autextification_dataset,
    load_autextification_multiazter_features,
    load_autextification_pucp_features,
)

LABEL_HUMAN = 0
LABEL_GENERATED = 1
SAMPLE_SIZE = 50


class RobertaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = load_model(
            "./models/autextification_roberta_cls_ft.h5",
            custom_objects={"TFRobertaModel": TFRobertaModel},
        )
        self.tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts, return_tensors="tf", padding=True, max_length=128, truncation=True
        )

    def predict_proba(self, X):
        x_tokenized = self.tokenize(X)
        return self.model.predict(
            {
                "input_ids": x_tokenized["input_ids"],
                "attention_mask": x_tokenized["attention_mask"],
            }
        )

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        return (pred_proba > 0.5).astype(int)


def train_model(
    repository_name: str,
    train_texts: list[str],
    test_texts: list[str],
    train_features: np.ndarray,
    train_labels: [int],
    test_features: np.ndarray,
    test_labels: [int],
):
    roberta_model = RobertaClassifier()
    xgb_pipeline = Pipeline([("clf", XGBClassifier())])
    xgb_parameters = {
        "clf__max_depth": range(1, 10, 3),
        "clf__n_estimators": range(20, 300, 50),
        "clf__learning_rate": [0.1, 0.01, 0.3],
    }

    xgb_model = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_parameters,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    models = [
        ("xgb", xgb_model),
    ]

    print("Training XGBoost using Roberta logits and PUCP features...")

    train_features, val_features, train_labels, val_labels, train_texts, val_texts = (
        train_test_split(
            train_features, train_labels, train_texts, test_size=0.20, random_state=42
        )
    )

    roberta_train_proba = roberta_model.predict_proba(train_texts)
    roberta_val_proba = roberta_model.predict_proba(val_texts)
    roberta_test_proba = roberta_model.predict_proba(test_texts)

    train_features = np.concatenate([train_features, roberta_train_proba], axis=1)
    val_features = np.concatenate([val_features, roberta_val_proba], axis=1)
    test_features = np.concatenate([test_features, roberta_test_proba], axis=1)

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

        # joblib.dump(model, f"./models/{model_name}_{repository_name}.joblib")

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
    training_output.to_csv(
        f"./results/autextification_roberta_proba_{repository_name}_{model_name}.csv"
    )

    print(f"Training results for {repository_name}:")
    print(training_output.head())


def train_ml_models():
    _, train_labels, _, test_labels = load_autextification_dataset()
    train_multiazter_features, test_multiazter_features = (
        load_autextification_multiazter_features()
    )
    train_cohmetrix_features, test_cohmetrix_features = (
        load_autextification_cohmetrix_features()
    )
    train_pucpmetrix_features, test_pucpmetrix_features = (
        load_autextification_pucp_features()
    )

    print("Multiazter features shape:", train_multiazter_features.shape)
    print("Multiazter test features shape:", test_multiazter_features.shape)
    print("Cohmetrix train features shape:", train_cohmetrix_features.shape)
    print("Cohmetrix test features shape:", test_cohmetrix_features.shape)
    print("PUCPMetrix features shape:", train_pucpmetrix_features.shape)
    print("PUCPMetrix test features shape:", test_pucpmetrix_features.shape)

    # train_model(
    #     "multiazter",
    #     train_multiazter_features,
    #     train_labels,
    #     test_multiazter_features,
    #     test_labels,
    # )
    #
    # train_model(
    #     "coh_metrix",
    #     train_cohmetrix_features,
    #     train_labels,
    #     test_cohmetrix_features,
    #     test_labels,
    # )
    train_model(
        "pucp",
        train_pucpmetrix_features,
        train_labels,
        test_pucpmetrix_features,
        test_labels,
    )


if __name__ == "__main__":
    train_ml_models()
