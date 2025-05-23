import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight, shuffle
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, AdamW
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TFAutoModelForSequenceClassification,
    TFRobertaForSequenceClassification,
    TFRobertaModel,
    Trainer,
    TrainingArguments,
)
from xgboost import XGBClassifier

from common import get_cohmetrix_dataset_grouped, get_multiazter_dataset_grouped
from datasets import load_dataset


def train_roberta_metrics_model(train_metrics: np.ndarray, test_metrics: np.ndarray):
    assert (
        train_metrics.shape[1] == test_metrics.shape[1]
    ), "Train and test metrics must have the same shape"

    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    x_train = np.array(train_dataset["text"])
    x_test = np.array(test_dataset["text"])

    assert (
        x_train.shape[0] == train_metrics.shape[0]
    ), "Texts and metrics must have the same number of samples"

    assert (
        x_test.shape[0] == test_metrics.shape[0]
    ), "Texts and metrics must have the same number of samples"

    y_train = np.array(train_dataset["label"])
    y_test = np.array(test_dataset["label"])

    x_train, x_val, y_train, y_val, train_metrics, val_metrics = train_test_split(
        x_train, y_train, train_metrics, test_size=0.20, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("echovl/roberta-bne-autex")

    def tokenize(texts: list[str]):
        return tokenizer(
            texts, return_tensors="tf", padding=True, max_length=128, truncation=True
        )

    x_train_tokenized = tokenize(list(x_train))
    x_val_tokenized = tokenize(list(x_val))
    x_test_tokenized = tokenize(list(x_test))

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    best_model = None
    best_val_score = 0
    scores = []

    # Train the model 5 times
    for run in range(5):
        print(f"Training run {run + 1}/5")

        input_ids = tf.keras.layers.Input(
            shape=(128,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(128,), dtype=tf.int32, name="attention_mask"
        )
        metrics = tf.keras.layers.Input(
            shape=(train_metrics.shape[1],), dtype=tf.float32, name="metrics"
        )

        roberta_model = TFRobertaModel.from_pretrained(
            "echovl/roberta-bne-autex",
            output_hidden_states=True,
        )

        roberta_model.trainable = False

        outputs = roberta_model(input_ids, attention_mask=attention_mask)
        cls_output = outputs.hidden_states[-1][:, 0, :]

        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(train_metrics)
        metrics_norm = normalizer(metrics)

        # cls_output = tf.keras.layers.Dropout(0.3)(cls_output)
        metrics_norm = tf.keras.layers.Dense(768, activation="relu")(
            metrics_norm
        )

        shared = tf.keras.layers.Dense(64)
        metrics_norm = shared(metrics_norm)
        cls_output = shared(cls_output)

        x = tf.keras.layers.Concatenate()([cls_output, metrics_norm])
        output = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(
            inputs=[input_ids, attention_mask, metrics], outputs=output
        )

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            metrics=["accuracy"],
            loss="binary_crossentropy",
        )

        model.fit(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
                "metrics": train_metrics,
            },
            y_train,
            validation_data=(
                {
                    "input_ids": x_val_tokenized["input_ids"],
                    "attention_mask": x_val_tokenized["attention_mask"],
                    "metrics": val_metrics,
                },
                y_val,
            ),
            epochs=10,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping],
        )

        train_pred = model.predict(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
                "metrics": train_metrics,
            }
        )
        train_output = (train_pred > 0.5).astype(int)
        train_score = f1_score(y_train, train_output, average="macro")

        val_pred = model.predict(
            {
                "input_ids": x_val_tokenized["input_ids"],
                "attention_mask": x_val_tokenized["attention_mask"],
                "metrics": val_metrics,
            }
        )
        val_output = (val_pred > 0.5).astype(int)
        val_score = f1_score(y_val, val_output, average="macro")

        test_pred = model.predict(
            {
                "input_ids": x_test_tokenized["input_ids"],
                "attention_mask": x_test_tokenized["attention_mask"],
                "metrics": test_metrics,
            }
        )
        test_output = (test_pred > 0.5).astype(int)
        test_score = f1_score(y_test, test_output, average="macro")

        print(f"Training F1 Score for run {run + 1}: {train_score}")
        print(f"Validation F1 Score for run {run + 1}: {val_score}")
        print(f"Test F1 Score for run {run + 1}: {test_score}")

        scores.append({"train": train_score, "val": val_score, "test": test_score})

        # Check if this is the best model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = roberta_model

    for score in scores:
        print(
            f"Run #{run + 1}: Train: {score['train']:.4f}, Val: {score['val']:.4f}, Test: {score['test']:.4f}"
        )

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv("./results/autextification_roberta_pucp.csv", index=False)

    # Push the best model to Hugging Face Hub
    if best_model is not None:
        model_name = "bertin-roberta-spanish-autotextification-best"
        # tokenizer.push_to_hub(model_name)
        # best_model.push_to_hub(model_name)
        print(f"Best model pushed to Hugging Face Hub: {model_name}")


def train_roberta_model():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    x_train = np.array(train_dataset["text"])
    x_test = np.array(test_dataset["text"])

    y_train = np.array(train_dataset["label"])
    y_test = np.array(test_dataset["label"])

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

    def tokenize(texts: list[str]):
        return tokenizer(
            texts, return_tensors="tf", padding=True, max_length=128, truncation=True
        )

    x_train_tokenized = tokenize(list(x_train))
    x_val_tokenized = tokenize(list(x_val))
    x_test_tokenized = tokenize(list(x_test))

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    best_model = None
    best_val_score = 0
    scores = []

    # Train the model 5 times
    for run in range(5):
        print(f"Training run {run + 1}/10")

        input_ids = tf.keras.layers.Input(
            shape=(128,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(128,), dtype=tf.int32, name="attention_mask"
        )

        roberta_model = TFRobertaModel.from_pretrained(
            "PlanTL-GOB-ES/roberta-base-bne",
            from_pt=True,
            output_hidden_states=True,
        )

        outputs = roberta_model(input_ids, attention_mask=attention_mask)
        cls_output = outputs.hidden_states[-1][:, 0, :]

        x = tf.keras.layers.Dropout(0.3)(cls_output)
        x = tf.keras.layers.Dense(768, activation="relu")(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=3e-5),
            metrics=["accuracy"],
            loss="binary_crossentropy",
        )

        # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        model.fit(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
            },
            y_train,
            validation_data=(
                {
                    "input_ids": x_val_tokenized["input_ids"],
                    "attention_mask": x_val_tokenized["attention_mask"],
                },
                y_val,
            ),
            epochs=5,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping],
            # class_weight=class_weights,
        )

        train_pred = model.predict(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
            }
        )
        train_output = (train_pred > 0.5).astype(int)
        train_score = f1_score(y_train, train_output, average="macro")

        val_pred = model.predict(
            {
                "input_ids": x_val_tokenized["input_ids"],
                "attention_mask": x_val_tokenized["attention_mask"],
            }
        )
        val_output = (val_pred > 0.5).astype(int)
        val_score = f1_score(y_val, val_output, average="macro")

        test_pred = model.predict(
            {
                "input_ids": x_test_tokenized["input_ids"],
                "attention_mask": x_test_tokenized["attention_mask"],
            }
        )
        test_output = (test_pred > 0.5).astype(int)
        test_score = f1_score(y_test, test_output, average="macro")

        print(f"Training F1 Score for run {run + 1}: {train_score}")
        print(f"Validation F1 Score for run {run + 1}: {val_score}")
        print(f"Test F1 Score for run {run + 1}: {test_score}")

        scores.append({"train": train_score, "val": val_score, "test": test_score})

        # Check if this is the best model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = roberta_model

    for score in scores:
        print(
            f"Run #{run + 1}: Train: {score['train']:.4f}, Val: {score['val']:.4f}, Test: {score['test']:.4f}"
        )

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(
        "./results/autextification_roberta_bne_finetuning.csv", index=False
    )

    # Push the best model to Hugging Face Hub
    if best_model is not None:
        model_name = "roberta-bne-autex"
        tokenizer.push_to_hub(model_name)
        best_model.push_to_hub(model_name)
        print(f"Best model pushed to Hugging Face Hub: {model_name}")


def validate_berta_model():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "echovl/bertin-roberta-spanish-autotextification"
    )
    tokenized_data = tokenizer(test_dataset["text"], return_tensors="np", padding=True)
    train_tokenized_data = tokenizer(
        train_dataset["text"], return_tensors="np", padding=True
    )

    # Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
    tokenized_data = dict(tokenized_data)
    train_tokenized_data = dict(train_tokenized_data)

    model = TFRobertaForSequenceClassification.from_pretrained(
        "echovl/bertin-roberta-spanish-autotextification"
    )

    test_labels = np.array(test_dataset["label"])
    train_labels = np.array(train_dataset["label"])

    test_output_logits = model.predict(tokenized_data).logits
    test_output = tf.math.argmax(test_output_logits, axis=-1)

    train_output_logits = model.predict(train_tokenized_data).logits
    train_output = tf.math.argmax(train_output_logits, axis=-1)

    test_score = f1_score(test_labels, test_output, average="macro")
    train_score = f1_score(train_labels, train_output, average="macro")

    print("Training BERTA score", train_score)
    print("Testing BERTA score", test_score)


def get_berta_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained(
        "echovl/bertin-roberta-spanish-autotextification"
    )
    model = TFRobertaModel.from_pretrained(
        "echovl/bertin-roberta-spanish-autotextification"
    )

    batch_size = 1024
    train_roberta_features = None
    for i in range(0, len(texts), batch_size):
        input = tokenizer(texts[i : i + batch_size], return_tensors="np", padding=True)
        input = dict(input)
        train_output = model.predict(input)
        cls_output = train_output.last_hidden_state[:, 0, :]

        print(cls_output.shape)
        if train_roberta_features is not None:
            train_roberta_features = np.concatenate(
                (train_roberta_features, cls_output), axis=0
            )
        else:
            train_roberta_features = cls_output

    return train_roberta_features


def train_berta_multiazter_model():
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

    print("Training data size:", len(train_dataset))

    train_roberta_features = np.load("./data/berta_roberta_features.npy")
    test_roberta_features = np.load("./data/berta_roberta_test_features.npy")

    print("Rorberta features shape", train_roberta_features.shape)

    # np.save("./data/berta_roberta_features.npy", train_roberta_features)
    # np.save("./data/berta_roberta_test_features.npy", test_roberta_features)

    train_multiazter_features = train_multiazter_df.to_numpy()
    test_multiazter_features = test_multiazter_df.to_numpy()

    train_features = np.concatenate(
        (train_roberta_features, train_multiazter_features), axis=1
    )
    test_features = np.concatenate(
        (test_roberta_features, test_multiazter_features), axis=1
    )

    train_labels = [data["label"] for data in train_dataset]
    test_labels = [data["label"] for data in test_dataset]

    print("Rorberta features shape", train_roberta_features.shape)
    print("Multiazter features shape", train_multiazter_features.shape)
    print("Train features shape", train_features.shape)

    rf_pipeline = Pipeline(
        [("scaler", MinMaxScaler()), ("clf", RandomForestClassifier())]
    )
    rf_parameters = {
        "clf__n_estimators": range(20, 250, 10),
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_features": ["sqrt", "log2"],
        "clf__max_depth": range(1, 3, 1),
    }

    xgb_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", XGBClassifier())])
    xgb_parameters = {
        "clf__max_depth": range(1, 3, 1),
        "clf__n_estimators": range(20, 250, 10),
        "clf__learning_rate": [0.1, 0.01, 0.05],
    }

    svc_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", LinearSVC())])
    svc_parameters = {
        "clf__C": range(1, 15, 2),
        "clf__penalty": ["l1", "l2"],
        "clf__dual": [False],
        "clf__max_iter": [40000],
    }

    rf_model = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_parameters,
        n_iter=10,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    xgb_model = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=xgb_parameters,
        n_iter=10,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    svc_model = RandomizedSearchCV(
        estimator=svc_pipeline,
        param_distributions=svc_parameters,
        n_iter=10,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    models = [("xgb", xgb_model), ("rf", rf_model), ("svc", svc_model)]

    for model_name, model in models:
        print(f"Training {model_name} model")

        X, y = shuffle(train_features, train_labels, random_state=42)

        model.fit(X, y)

        joblib.dump(model, f"./models/berta_multiazter_{model_name}.pkl", compress=1)

        train_output = model.predict(train_features)
        test_output = model.predict(test_features)

        train_score = f1_score(train_labels, train_output, average="macro")
        test_score = f1_score(test_labels, test_output, average="macro")

        print(f"Training {model_name} score", train_score)
        print(f"Testing {model_name} score", test_score)

        print("CV best parameters: ", model.best_params_)
        print("CV best results: ", model.best_score_)


def train_berta_extended_model_keras(
    name: str, extra_train_features: np.ndarray, extra_test_features: np.ndarray
):
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    test_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="test"
    )

    train_roberta_features = np.load("./data/berta_roberta_features.npy")
    test_roberta_features = np.load("./data/berta_roberta_test_features.npy")

    train_features = (
        train_roberta_features
        if extra_train_features is None
        else np.concatenate((train_roberta_features, extra_train_features), axis=1)
    )
    test_features = (
        test_roberta_features
        if extra_test_features is None
        else np.concatenate((test_roberta_features, extra_test_features), axis=1)
    )

    train_labels = np.array([data["label"] for data in train_dataset])
    test_labels = np.array([data["label"] for data in test_dataset])

    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(train_features)

    model = keras.Sequential(
        [
            layers.Input(shape=(train_features.shape[1],)),
            normalizer,
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print(
        f"Model {name}: Training with {extra_train_features.shape[1] if extra_train_features is not None else 0} extra features..."
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    X, y = shuffle(train_features, train_labels)

    model.fit(
        X,
        y,
        validation_data=(test_features, test_labels),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )

    train_pred = model.predict(train_features)
    train_pred_labels = (train_pred > 0.5).astype(int)
    train_score = f1_score(train_labels, train_pred_labels, average="macro")

    test_pred = model.predict(test_features)
    test_pred_labels = (test_pred > 0.5).astype(int)
    test_score = f1_score(test_labels, test_pred_labels, average="macro")

    print(f"Model {name}: Train F1 score: {train_score:.4f}")
    print(f"Model {name}: Test F1 score: {test_score:.4f}")


def train_berta_multiazter_model_keras():
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

    print("Training data size:", len(train_dataset))

    train_roberta_features = np.load("./data/berta_roberta_features.npy")
    test_roberta_features = np.load("./data/berta_roberta_test_features.npy")

    print("Roberta features shape", train_roberta_features.shape)

    # np.save("./data/berta_roberta_features.npy", train_roberta_features)
    # np.save("./data/berta_roberta_test_features.npy", test_roberta_features)

    train_multiazter_features = train_multiazter_df.to_numpy()
    test_multiazter_features = test_multiazter_df.to_numpy()

    train_features = np.concatenate(
        (train_roberta_features, train_multiazter_features), axis=1
    )
    test_features = np.concatenate(
        (test_roberta_features, test_multiazter_features), axis=1
    )

    train_labels = np.array([data["label"] for data in train_dataset])
    test_labels = np.array([data["label"] for data in test_dataset])

    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(train_features)

    print("Roberta features shape", train_roberta_features.shape)
    print("Multiazter features shape", train_multiazter_features.shape)
    print("Train features shape", train_features.shape)

    model = keras.Sequential(
        [
            layers.Normalization(axis=-1),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        train_features,
        train_labels,
        validation_data=(test_features, test_labels),
        epochs=20,
        batch_size=32,
        verbose=1,
    )

    train_pred = model.predict(train_features)
    train_pred_labels = (train_pred > 0.5).astype(int)
    train_score = f1_score(train_labels, train_pred_labels, average="macro")

    test_pred = model.predict(test_features)
    test_pred_labels = (test_pred > 0.5).astype(int)
    test_score = f1_score(test_labels, test_pred_labels, average="macro")

    print(f"Train F1 score: {train_score:.4f}")
    print(f"Test F1 score: {test_score:.4f}")


def train_grouped_metrics_model():
    cohmetrix_grouped_dataset = get_cohmetrix_dataset_grouped()
    multiazter_grouped_dataset = get_multiazter_dataset_grouped()

    for group_name, dataset in cohmetrix_grouped_dataset.items():
        train_features = dataset["train_features"]
        test_features = dataset["test_features"]

        train_berta_extended_model_keras(
            f"coh_metrix_{group_name.lower()}", train_features, test_features
        )

    for group_name, dataset in multiazter_grouped_dataset.items():
        train_features = dataset["train_features"]
        test_features = dataset["test_features"]

        train_berta_extended_model_keras(
            f"multiazter_{group_name.lower()}", train_features, test_features
        )


def train_grouped_merged_metrics_model():
    cohmetrix_grouped_dataset = get_cohmetrix_dataset_grouped()
    multiazter_grouped_dataset = get_multiazter_dataset_grouped()

    cohmetrix_connectives_dataset = cohmetrix_grouped_dataset["connectives"]
    cohmetrix_descriptive_dataset = cohmetrix_grouped_dataset["descriptive"]
    cohmetrix_readability_dataset = cohmetrix_grouped_dataset["readability"]
    cohmetrix_word_information_dataset = cohmetrix_grouped_dataset["word information"]
    cohmetrix_lexical_diversity_dataset = cohmetrix_grouped_dataset["lexical diversity"]
    cohmetrix_referential_cohesion_dataset = cohmetrix_grouped_dataset[
        "referential cohesion"
    ]
    cohmetrix_syntactic_complexity_dataset = cohmetrix_grouped_dataset[
        "syntactic complexity"
    ]
    cohmetrix_syntactic_pattern_density_dataset = cohmetrix_grouped_dataset[
        "syntactic pattern density"
    ]

    multiazter_connectives_dataset = multiazter_grouped_dataset["connectives"]
    multiazter_descriptive_dataset = multiazter_grouped_dataset["descriptive"]
    multiazter_readability_dataset = multiazter_grouped_dataset["readability"]
    multiazter_word_frequency_dataset = multiazter_grouped_dataset["word frequency"]
    multiazter_lexical_diversity_dataset = multiazter_grouped_dataset[
        "lexical diversity"
    ]
    multiazter_referential_cohesion_dataset = multiazter_grouped_dataset[
        "referential cohesion"
    ]
    multiazter_word_semantic_information_dataset = multiazter_grouped_dataset[
        "word semantic information"
    ]
    multiazter_word_morphological_information_dataset = multiazter_grouped_dataset[
        "word morphological information"
    ]
    multiazter_syntactic_complexity_pattern_density_dataset = (
        multiazter_grouped_dataset["syntactic complexity & syntactic pattern density"]
    )

    merged_metrics_dataset = {
        "connectives": {
            "train_features": np.concatenate(
                (
                    multiazter_connectives_dataset["train_features"],
                    cohmetrix_connectives_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_connectives_dataset["test_features"],
                    cohmetrix_connectives_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "descriptive": {
            "train_features": np.concatenate(
                (
                    multiazter_descriptive_dataset["train_features"],
                    cohmetrix_descriptive_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_descriptive_dataset["test_features"],
                    cohmetrix_descriptive_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "readability": {
            "train_features": np.concatenate(
                (
                    multiazter_readability_dataset["train_features"],
                    cohmetrix_readability_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_readability_dataset["test_features"],
                    cohmetrix_readability_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "word information": {
            "train_features": np.concatenate(
                (
                    multiazter_word_semantic_information_dataset["train_features"],
                    multiazter_word_morphological_information_dataset["train_features"],
                    multiazter_word_frequency_dataset["train_features"],
                    cohmetrix_word_information_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_word_semantic_information_dataset["test_features"],
                    multiazter_word_morphological_information_dataset["test_features"],
                    multiazter_word_frequency_dataset["test_features"],
                    cohmetrix_word_information_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "lexical diversity": {
            "train_features": np.concatenate(
                (
                    multiazter_lexical_diversity_dataset["train_features"],
                    cohmetrix_lexical_diversity_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_lexical_diversity_dataset["test_features"],
                    cohmetrix_lexical_diversity_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "referential cohesion": {
            "train_features": np.concatenate(
                (
                    multiazter_referential_cohesion_dataset["train_features"],
                    cohmetrix_referential_cohesion_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_referential_cohesion_dataset["test_features"],
                    cohmetrix_referential_cohesion_dataset["test_features"],
                ),
                axis=1,
            ),
        },
        "syntactic complexity & syntactic pattern density": {
            "train_features": np.concatenate(
                (
                    multiazter_syntactic_complexity_pattern_density_dataset[
                        "train_features"
                    ],
                    cohmetrix_syntactic_complexity_dataset["train_features"],
                    cohmetrix_syntactic_pattern_density_dataset["train_features"],
                ),
                axis=1,
            ),
            "test_features": np.concatenate(
                (
                    multiazter_syntactic_complexity_pattern_density_dataset[
                        "test_features"
                    ],
                    cohmetrix_syntactic_complexity_dataset["test_features"],
                    cohmetrix_syntactic_pattern_density_dataset["test_features"],
                ),
                axis=1,
            ),
        },
    }

    for group_name, dataset in merged_metrics_dataset.items():
        train_features = dataset["train_features"]
        test_features = dataset["test_features"]

        train_berta_extended_model_keras(
            f"merged_{group_name.lower()}", train_features, test_features
        )


def train_berta_pucp_model():
    train_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_train_pucp_indicators.csv", index_col="index"
    )
    test_pucpmetrix_df = pd.read_csv(
        "./datasets/autextification_test_pucp_indicators.csv", index_col="index"
    )

    print("Train pucp metrics shape:", train_pucpmetrix_df.to_numpy().shape)
    print("Test pucp metrics shape:", test_pucpmetrix_df.to_numpy().shape)

    train_roberta_metrics_model(
        train_pucpmetrix_df.to_numpy(),
        test_pucpmetrix_df.to_numpy(),
    )


if __name__ == "__main__":
    # train_berta_extended_model_keras("baseline", None, None)
    # train_grouped_metrics_model()
    # train_roberta_bne_model()
    # train_grouped_merged_metrics_model()
    train_berta_pucp_model()
    # train_roberta_model()
