import joblib
import numpy as np
import pandas as pd
from common import get_cohmetrix_dataset_grouped, get_multiazter_dataset_grouped
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier


def train_model(
    repository_name: str,
    train_features: np.ndarray,
    train_labels: [int],
    test_features: np.ndarray,
    test_labels: [int],
):
    xgb_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", XGBClassifier())])
    xgb_parameters = {
        "clf__max_depth": range(2, 10, 1),
        "clf__n_estimators": range(50, 250, 50),
        "clf__learning_rate": [0.1, 0.01, 0.05],
    }

    svc_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", LinearSVC())])
    svc_parameters = {
        "clf__C": range(1, 15, 2),
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [False],
        "clf__max_iter": [40000],
    }

    lr_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression())])
    lr_parameters = {
        "clf__C": range(1, 15, 2),
        "clf__dual": [False],
        "clf__max_iter": [20000],
    }

    xgb_model = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_parameters,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    svc_model = GridSearchCV(
        estimator=svc_pipeline,
        param_grid=svc_parameters,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    lr_model = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=lr_parameters,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    models = [("xgb", xgb_model), ("svm", svc_model), ("lr", lr_model)]

    print(f"Processing {repository_name}")

    train_scores = []
    test_scores = []

    for model_name, model in models:
        print(f"Training {model_name} model")

        X, y = shuffle(train_features, train_labels, random_state=42)

        model.fit(X, y)

        joblib.dump(model, f"./models/{repository_name}_{model_name}.pkl", compress=1)

        train_output = model.predict(train_features)
        test_output = model.predict(test_features)

        train_score = f1_score(train_labels, train_output, average="macro")
        test_score = f1_score(test_labels, test_output, average="macro")

        train_scores.append(train_score)
        test_scores.append(test_score)

        print(f"Training {model_name} score", train_score)
        print(f"Testing {model_name} score", test_score)

    training_output = pd.DataFrame(
        {
            "model": [model_name for model_name, _ in models],
            "train_score": train_scores,
            "test_score": test_scores,
        }
    )
    training_output.to_csv(f"./data/{repository_name}_training_output.csv")


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

    train_multiazter_features = train_multiazter_df.to_numpy()
    test_multiazter_features = test_multiazter_df.to_numpy()
    train_cohmetrix_features = train_cohmetrix_df.to_numpy()
    test_cohmetrix_features = test_cohmetrix_df.to_numpy()

    print("Multiazter features shape:", train_multiazter_features.shape)
    print("Cohmetrix features shape:", train_cohmetrix_features.shape)

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


def train_ml_models_grouped():
    # cross validation using scikit-learn
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    # train_multiazter_df = pd.read_csv(
    #     "./data/train_multiazter_metrics.csv", index_col="index"
    # )
    # test_multiazter_df = pd.read_csv(
    #     "./data/test_multiazter_metrics.csv", index_col="index"
    # )

    train_labels = [data["label"] for data in train_dataset]
    test_labels = [data["label"] for data in test_dataset]

    cohmetrix_grouped_dataset = get_cohmetrix_dataset_grouped()
    multiazter_grouped_dataset = get_multiazter_dataset_grouped()

    print(multiazter_grouped_dataset)

    for group_name, dataset in cohmetrix_grouped_dataset.items():
        train_features = dataset["train_features"]
        test_features = dataset["test_features"]

        train_model(
            f"coh_metrix_{group_name.lower()}",
            train_features,
            train_labels,
            test_features,
            test_labels,
        )

    for group_name, dataset in multiazter_grouped_dataset.items():
        train_features = dataset["train_features"]
        test_features = dataset["test_features"]

        train_model(
            f"multiazter_{group_name.lower()}",
            train_features,
            train_labels,
            test_features,
            test_labels,
        )


if __name__ == "__main__":
    train_ml_models_grouped()
