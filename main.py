import json
import math
import pathlib
import subprocess
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer
from text_complexity_analyzer_cm.utils.utils import preprocess_text_spanish
from transformers import AutoModelForMaskedLM, AutoTokenizer
from xgboost import XGBClassifier

MULTIAZTER_PYTHON_PATH = "/home/echovl/MultiAzterTest/.venv/bin/python"
MULTIAZTER_NUM_WORKERS = 1

LABEL_HUMAN = 0
LABEL_GENERATED = 1
SAMPLE_SIZE = 50

tca = TextComplexityAnalyzer("es", preprocessing_func=preprocess_text_spanish)


def coh_metrix_metrics(texts: [str]) -> List[OrderedDict[str, float]]:
    metrics = tca.calculate_all_indices_for_texts(texts, workers=16, batch_size=1)

    # replace None values with 0
    for m in metrics:
        for k, v in m.items():
            if v is None:
                m[k] = 0

    return [OrderedDict(m) for m in metrics]


def multiazter_metrics_batch(
    texts: [str], language: str = "spanish"
) -> List[OrderedDict[str, float]]:
    num_texts = len(texts)
    batch_size = math.ceil(num_texts / MULTIAZTER_NUM_WORKERS)
    texts_batches = [texts[i : i + batch_size] for i in range(0, num_texts, batch_size)]

    print("Number of batches: ", len(texts_batches))

    metrics = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(multiazter_metrics, texts_batches)
        for result in results:
            metrics.extend(result)

    return metrics


def multiazter_metrics(
    texts: [str], language: str = "spanish"
) -> List[OrderedDict[str, float]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, text in enumerate(texts):
            with open(f"{tmpdir}/{i}.txt", "w") as f:
                f.write(text)

        texts_path = [f"{tmpdir}/{i}.txt" for i, _ in enumerate(texts)]

        print("Running multiazter")

        p = subprocess.run(
            [
                MULTIAZTER_PYTHON_PATH,
                "multiaztertest.py",
                "-c",
                "-r",
                "-f",
                *texts_path,
                "-l",
                "spanish",
                "-m",
                "stanford",
                "-d",
                "./workdir",
            ],
            cwd=f"{pathlib.Path().resolve()}/../MultiAzterTest",
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
            capture_output=True,
            text=True,
        )

        print("Multiazter finished")

        raw_result = p.stdout.split("__MULTIAZTER_RESULT__")[1]
        metrics = json.loads(raw_result)

        # replace None values with 0
        for m in metrics:
            for k, v in m.items():
                if v is None:
                    m[k] = 0

        return [OrderedDict(m) for m in metrics]


def compute_and_save_metrics():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )

    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_texts = [data["text"] for data in train_dataset]
    test_texts = [data["text"] for data in test_dataset]

    train_multiazter_metrics = multiazter_metrics_batch(train_texts, language="spanish")
    test_multiazter_metrics = multiazter_metrics_batch(test_texts, language="spanish")

    train_coh_metrix_metrics = coh_metrix_metrics(train_texts)
    test_coh_metrix_metrics = coh_metrix_metrics(test_texts)

    train_multiazter_df = pd.DataFrame(train_multiazter_metrics)
    train_multiazter_df.to_csv("train_multiazter_metrics.csv", index_label="index")

    test_multiazter_df = pd.DataFrame(test_multiazter_metrics)
    test_multiazter_df.to_csv("test_multiazter_metrics.csv", index_label="index")

    train_coh_metrix_df = pd.DataFrame(train_coh_metrix_metrics)
    train_coh_metrix_df.to_csv("train_coh_metrix_metrics.csv", index_label="index")

    test_coh_metrix_df = pd.DataFrame(test_coh_metrix_metrics)
    test_coh_metrix_df.to_csv("test_coh_metrix_metrics.csv", index_label="index")


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

    for model_name, model in models:
        print(f"Training {model_name} model")

        X, y = shuffle(train_features, train_labels, random_state=42)

        model.fit(X, y)

        joblib.dump(model, f"./models/{repository_name}_{model_name}.pkl", compress=1)

        train_score = model.score(train_features, train_labels)
        test_score = model.score(test_features, test_labels)

        print(f"Training {model_name} score", train_score)
        print(f"Testing {model_name} score", test_score)


def train_berta_model():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )
    train_texts = [data["text"] for data in train_dataset]

    tokenizer = AutoTokenizer.from_pretrained(
        "bertin-project/bertin-roberta-base-spanish"
    )

    print(tokenizer(train_texts[0]))


def main():
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


train_berta_model()
