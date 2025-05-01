from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from iapucp_metrix.analyzer import Analyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

LABEL_HUMAN = 0
LABEL_GENERATED = 1
SAMPLE_SIZE = 50


pucp_metrix = Analyzer()


def pucp_metrics(texts: [str]) -> list[OrderedDict[str, float]]:
    metrics = pucp_metrix.compute_metrics(texts, workers=16, batch_size=100)

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
    test_texts = [data["text"] for data in test_dataset]

    print("Number of Train texts:", len(train_texts))
    print("Number of Test texts:", len(test_texts))

    train_pucp_metrics = pucp_metrics(train_texts)
    test_pucp_metrics = pucp_metrics(test_texts)

    train_pucp_df = pd.DataFrame(train_pucp_metrics)
    train_pucp_df.to_csv("train_pucp_metrics.csv", index_label="index")

    test_pucp_df = pd.DataFrame(test_pucp_metrics)
    test_pucp_df.to_csv("test_pucp_metrics.csv", index_label="index")

    # train_multiazter_metrics = multiazter_metrics_batch(train_texts, language="spanish")
    # test_multiazter_metrics = multiazter_metrics_batch(test_texts, language="spanish")
    #
    # train_coh_metrix_metrics = coh_metrix_metrics(train_texts)
    # test_coh_metrix_metrics = coh_metrix_metrics(test_texts)
    #
    # train_multiazter_df = pd.DataFrame(train_multiazter_metrics)
    # train_multiazter_df.to_csv("train_multiazter_metrics.csv", index_label="index")
    #
    # test_multiazter_df = pd.DataFrame(test_multiazter_metrics)
    # test_multiazter_df.to_csv("test_multiazter_metrics.csv", index_label="index")
    #
    # train_coh_metrix_df = pd.DataFrame(train_coh_metrix_metrics)
    # train_coh_metrix_df.to_csv("train_coh_metrix_metrics.csv", index_label="index")

    # test_coh_metrix_df = pd.DataFrame(test_coh_metrix_metrics)
    # test_coh_metrix_df.to_csv("test_coh_metrix_metrics.csv", index_label="index")


def train_model(
    repository_name: str,
    train_features: np.ndarray,
    train_labels: [int],
    test_features: np.ndarray,
    test_labels: [int],
):
    xgb_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier())])
    xgb_parameters = {
        "clf__max_depth": range(1, 30, 5),
        "clf__n_estimators": range(20, 250, 25),
        "clf__learning_rate": [0.1, 0.01, 0.05],
    }

    svc_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC())])
    svc_parameters = {
        "clf__C": range(1, 15, 2),
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [False],
        "clf__max_iter": [40000],
    }

    lr_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    )
    lr_parameters = {
        "clf__C": range(1, 15, 2),
        "clf__dual": [False],
        "clf__max_iter": [20000],
    }
    rf_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", RandomForestClassifier())]
    )
    rf_parameters = {
        "clf__n_estimators": range(20, 250, 25),
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_features": ["sqrt", "log2"],
        "clf__max_depth": range(1, 30, 5),
    }

    xgb_model = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=xgb_parameters,
        scoring="f1_macro",
        n_iter=10,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    svc_model = RandomizedSearchCV(
        estimator=svc_pipeline,
        param_distributions=svc_parameters,
        scoring="f1_macro",
        n_iter=10,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    lr_model = RandomizedSearchCV(
        estimator=lr_pipeline,
        param_distributions=lr_parameters,
        scoring="f1_macro",
        n_iter=10,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    rf_model = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_parameters,
        scoring="f1_macro",
        n_iter=10,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    models = [
        ("xgb", xgb_model),
        ("svm", svc_model),
        ("lr", lr_model),
        ("rf", rf_model),
    ]

    print(f"Processing {repository_name}")
    train_scores = []
    test_scores = []

    for model_name, model in models:
        print(f"Training {model_name} model")

        X, y = shuffle(train_features, train_labels, random_state=42)

        model.fit(X, y)

        # joblib.dump(model, f"./models/{repository_name}_{model_name}.pkl", compress=1)

        train_output = model.predict(train_features)
        test_output = model.predict(test_features)

        train_score = f1_score(train_labels, train_output, average="macro")
        test_score = f1_score(test_labels, test_output, average="macro")

        train_scores.append(train_score)
        test_scores.append(test_score)

        print(f"Training {model_name} score", train_score)
        print(f"Testing {model_name} score", test_score)

        print("CV best parameters: ", model.best_params_)
        print("CV best results: ", model.best_score_)

    training_output = pd.DataFrame(
        {
            "model": [model_name for model_name, _ in models],
            "train_score": train_scores,
            "test_score": test_scores,
        }
    )
    training_output.to_csv(f"./results/{repository_name}_training_output.csv")


def train_ml_models():
    # cross validation using scikit-learn
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_multiazter_df = pd.read_csv(
        "./data/train_multiazter_metrics.csv", index_col="index"
    )
    test_multiazter_df = pd.read_csv(
        "./data/test_multiazter_metrics.csv", index_col="index"
    )
    train_cohmetrix_df = pd.read_csv(
        "./data/train_coh_metrix_metrics.csv", index_col="index"
    )
    test_cohmetrix_df = pd.read_csv(
        "./data/test_coh_metrix_metrics.csv", index_col="index"
    )
    train_pucpmetrix_df = pd.read_csv(
        "./data/train_pucp_metrics.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv("./data/test_pucp_metrics.csv", index_col="index")

    train_multiazter_features = train_multiazter_df.to_numpy()
    test_multiazter_features = test_multiazter_df.to_numpy()
    train_cohmetrix_features = train_cohmetrix_df.to_numpy()
    test_cohmetrix_features = test_cohmetrix_df.to_numpy()
    train_pucpmetrix_features = train_pucpmetrix_df.to_numpy()
    test_pucpmetrix_features = test_pucpmetrix_df.to_numpy()

    print("Multiazter features shape:", train_multiazter_features.shape)
    print("Cohmetrix features shape:", train_cohmetrix_features.shape)
    print("PUCPMetrix features shape:", train_pucpmetrix_features.shape)

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


if __name__ == "__main__":
    # train_berta_multiazter_model()
    # compute_and_save_pucp_metrics()
    train_ml_models()
