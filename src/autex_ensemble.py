import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from collections import OrderedDict

import numpy as np
import pandas as pd
from iapucp_metrix.analyzer import Analyzer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFRobertaModel
from xgboost import XGBClassifier

from dataloader import load_autextification_dataset, load_autextification_pucp_features
from src.common import compute_evaluation_scores, merge_scores

pucp_metrix = Analyzer()


def pucp_metrics(texts: [str]) -> list[OrderedDict[str, float]]:
    metrics = pucp_metrix.compute_metrics(texts, workers=1, batch_size=512)

    # replace None values with 0
    for m in metrics:
        for k, v in m.items():
            if v is None:
                m[k] = 0

    return [OrderedDict(m) for m in metrics]


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = XGBClassifier()
        self.model.load_model("./models/xgboost_pucp_model.json")

    def fit(self, X, y=None):
        return self

    def compute_metrics(self, X, y=None):
        features = pucp_metrics(X)
        df = pd.DataFrame(features)
        return df.to_numpy()

    def predict_proba(self, X):
        return self.model.predict_proba(self.compute_metrics(X))

    def predict(self, X):
        return self.model.predict(self.compute_metrics(X))


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


def train_ensemble_classifier():
    train_texts, train_labels, test_texts, test_labels = load_autextification_dataset()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.20, random_state=42
    )

    roberta_model = RobertaClassifier()
    xgboost_model = XGBoostClassifier()
    estimators = [("roberta", roberta_model), ("xgboost", xgboost_model)]

    # You can set "stack_method="predict_proba" for more control
    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=2,
        passthrough=False,
        verbose=1,
    )

    stacking_classifier.fit(train_texts, train_labels)

    train_predicted = stacking_classifier.predict(train_texts)
    val_predicted = stacking_classifier.predict(val_texts)
    test_predicted = stacking_classifier.predict(test_texts)

    train_scores = compute_evaluation_scores(train_labels, train_predicted)
    val_scores = compute_evaluation_scores(val_labels, val_predicted)
    test_scores = compute_evaluation_scores(test_labels, test_predicted)

    scores = pd.DataFrame(
        merge_scores(train_scores, val_scores, test_scores),
        columns=["train", "val", "test"],
    )
    scores.to_csv("./results/autextification_ensemble.csv")


def evaluate_xgboost():
    train_texts, train_labels, test_texts, test_labels = load_autextification_dataset()
    train_pucpmetrix_features, test_pucpmetrix_features = (
        load_autextification_pucp_features()
    )
    test_pucpmetrix_df = pd.DataFrame(pucp_metrics(test_texts))
    test_pucpmetrix_features_np = test_pucpmetrix_df.to_numpy()

    print("PUCPMetrix features shape:", train_pucpmetrix_features.shape)
    print("PUCPMetrix test features shape:", test_pucpmetrix_features.shape)

    xgboost_model = XGBClassifier()
    xgboost_model.load_model("./models/xgboost_pucp_model.json")

    train_predicted = xgboost_model.predict(train_pucpmetrix_features)
    test_predicted = xgboost_model.predict(test_pucpmetrix_features)
    test_pred_2 = xgboost_model.predict(test_pucpmetrix_features_np)

    train_score = f1_score(train_labels, train_predicted, average="macro")
    test_score = f1_score(test_labels, test_predicted, average="macro")
    test_pred = f1_score(test_labels, test_pred_2, average="macro")

    print(f"Train score: {train_score}, Test score: {test_score}")
    print(f"Test score 2: {test_pred}")


if __name__ == "__main__":
    train_ensemble_classifier()
