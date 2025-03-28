import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification,
                          TFRobertaForSequenceClassification, TFRobertaModel)
from xgboost import XGBClassifier


def train_berta_model():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    # test_dataset = load_dataset(
    #     "symanto/autextification2023", "detection_es", split="test"
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        "bertin-project/bertin-roberta-base-spanish"
    )
    tokenized_data = tokenizer(train_dataset["text"], return_tensors="np", padding=True)

    # Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
    tokenized_data = dict(tokenized_data)

    labels = np.array(train_dataset["label"])  # Label is already an array of 0 and 1

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bertin-project/bertin-roberta-base-spanish"
    )

    # Lower learning rates are often better for fine-tuning transformers
    model.compile(optimizer=Adam(3e-5))  # No loss argument!
    model.fit(tokenized_data, labels)

    # Push models to Hugging Face Hub
    tokenizer.push_to_hub("bertin-roberta-spanish-autotextification")
    model.push_to_hub("bertin-roberta-spanish-autotextification")


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

    train_features = np.concatenate(
        (train_roberta_features, extra_train_features), axis=1
    )
    test_features = np.concatenate((test_roberta_features, extra_test_features), axis=1)

    train_labels = np.array([data["label"] for data in train_dataset])
    test_labels = np.array([data["label"] for data in test_dataset])

    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(train_features)

    model = keras.Sequential(
        [
            layers.Input(shape=(train_features.shape[1],)),
            normalizer,
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print(f"Model {name}: Training with {extra_train_features.shape[1]} extra features...")


    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_features,
        train_labels,
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

    print("Rorberta features shape", train_roberta_features.shape)
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

    print("Compiling model...")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Fitting model...")

    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(train_features)

    model = keras.Sequential(
        [
            layers.Input(shape=(train_features.shape[1],)),
            normalizer,
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

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


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Check for TensorFlow GPU access
print(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}"
)

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# train_berta_multiazter_model_keras()
