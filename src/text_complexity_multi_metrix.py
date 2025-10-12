import os
from pprint import pprint

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFRobertaModel
from xgboost import XGBClassifier

from dataloader import (
    load_text_complexity_multi_level_dataset,
    load_text_complexity_multiazter_features,
    load_text_complexity_pucp_features,
)

LABEL_SIMPLE = 1
LABEL_COMPLEX = 0


def create_roberta_model(learning_rate: float):
    input_ids = layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    roberta_model = TFRobertaModel.from_pretrained(
        "PlanTL-GOB-ES/roberta-base-bne",
        from_pt=True,
        output_hidden_states=True,
    )

    outputs = roberta_model(input_ids, attention_mask=attention_mask)
    cls_output = outputs.hidden_states[-1][:, 0, :]
    output = layers.Dense(3, activation="softmax")(cls_output)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
        loss="sparse_categorical_crossentropy",
    )

    return model, roberta_model


def train_roberta_model():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_complexity_multi_level_dataset()
    )

    lr = 3e-5
    epochs = 3
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

    def tokenize(texts: list[str]):
        return tokenizer(
            texts, return_tensors="tf", padding=True, max_length=128, truncation=True
        )

    x_train_tokenized = tokenize(list(train_texts))
    x_val_tokenized = tokenize(list(val_texts))
    x_test_tokenized = tokenize(list(test_texts))
    model, roberta_model = create_roberta_model(learning_rate=lr)
    history = model.fit(
        {
            "input_ids": x_train_tokenized["input_ids"],
            "attention_mask": x_train_tokenized["attention_mask"],
        },
        np.array(train_labels),
        validation_data=(
            {
                "input_ids": x_val_tokenized["input_ids"],
                "attention_mask": x_val_tokenized["attention_mask"],
            },
            np.array(val_labels),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    train_pred = model.predict(
        {
            "input_ids": x_train_tokenized["input_ids"],
            "attention_mask": x_train_tokenized["attention_mask"],
        }
    )
    train_output = np.argmax(train_pred, axis=1).astype(int)

    val_pred = model.predict(
        {
            "input_ids": x_val_tokenized["input_ids"],
            "attention_mask": x_val_tokenized["attention_mask"],
        }
    )
    # val_output = (val_pred > 0.5).astype(int)
    val_output = np.argmax(val_pred, axis=1).astype(int)

    test_pred = model.predict(
        {
            "input_ids": x_test_tokenized["input_ids"],
            "attention_mask": x_test_tokenized["attention_mask"],
        }
    )
    # test_output = (test_pred > 0.5).astype(int)
    test_output = np.argmax(test_pred, axis=1).astype(int)

    def get_scores(y_true, y_pred):
        f1_macro = f1_score(y_true, y_pred, average="macro")
        basic_f1, intermediate_f1, advanced_f1 = f1_score(y_true, y_pred, average=None)
        basic_rec, intermediate_rec, advanced_rec = recall_score(
            y_true, y_pred, average=None
        )
        basic_prec, intermediate_prec, advanced_prec = precision_score(
            y_true, y_pred, average=None
        )
        return {
            "f1_macro": f1_macro,
            "basic_f1": basic_f1,
            "intermediate_f1": intermediate_f1,
            "advanced_f1": advanced_f1,
            "basic_precision": basic_prec,
            "intermediate_precision": intermediate_prec,
            "advanced_precision": advanced_prec,
            "basic_recall": basic_rec,
            "intermediate_recall": intermediate_rec,
            "advanced_recall": advanced_rec,
        }

    domains = {
        "train": [train_labels, train_output],
        "test": [test_labels, test_output],
        "val": [val_labels, val_output],
    }

    model_results = {
        "model": ["roberta"],
    }

    for domain, data in domains.items():
        y_true, y_pred = data
        scores = get_scores(y_true, y_pred)

        print(f"Roberta {domain} scores: \n")
        pprint(scores)
        print("\n")

        for key, value in scores.items():
            result_key = f"{domain}_{key}"

            if result_key not in model_results:
                model_results[result_key] = []
            model_results[result_key].append(value)

    training_output = pd.DataFrame(model_results)
    training_output.to_csv("./results/text_complexity_multi_roberta_ml.csv")

    print("Training results for Roberta:")
    print(training_output.head())


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
        "clf__max_depth": range(1, 10, 3),
        "clf__n_estimators": range(50, 501, 100),
        "clf__learning_rate": [0.05, 0.005],
    }

    svc_pipeline = Pipeline([("scaler", RobustScaler()), ("clf", LinearSVC())])
    svc_parameters = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [False],
        "clf__max_iter": [40000],
    }

    lr_pipeline = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression())])
    lr_parameters = {
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__solver": ["lbfgs", "liblinear", "saga"],
        "clf__dual": [False],
        "clf__max_iter": [20000],
    }
    rf_pipeline = Pipeline([("clf", RandomForestClassifier())])
    rf_parameters = {
        "clf__n_estimators": [100, 300, 500, 700],
        "clf__max_features": ["sqrt", "log2"],
        "clf__max_depth": [3, 5, 10, 12, 15],
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
        basic_f1, intermediate_f1, advanced_f1 = f1_score(y_true, y_pred, average=None)
        basic_rec, intermediate_rec, advanced_rec = recall_score(
            y_true, y_pred, average=None
        )
        basic_prec, intermediate_prec, advanced_prec = precision_score(
            y_true, y_pred, average=None
        )
        return {
            "f1_macro": f1_macro,
            "basic_f1": basic_f1,
            "intermediate_f1": intermediate_f1,
            "advanced_f1": advanced_f1,
            "basic_precision": basic_prec,
            "intermediate_precision": intermediate_prec,
            "advanced_precision": advanced_prec,
            "basic_recall": basic_rec,
            "intermediate_recall": intermediate_rec,
            "advanced_recall": advanced_rec,
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
        print("Best model parameters:")
        pprint(model.best_params_)

    training_output = pd.DataFrame(model_results)
    training_output.to_csv(f"./results/text_complexity_multi_{repository_name}_ml.csv")

    print(f"Training results for {repository_name}:")
    print(
        training_output[
            ["model", "cv_f1_macro", "train_f1_macro", "val_f1_macro", "test_f1_macro"]
        ].head()
    )


def train_ml_models():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_complexity_multi_level_dataset()
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
    # train_roberta_model()
