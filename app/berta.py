import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFRobertaForSequenceClassification,
    TFRobertaModel,
)


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
        "clf__n_estimators": range(50, 1000, 50),
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_features": ["sqrt", "log2"],
        "clf__max_depth": range(2, 15, 2),
    }

    rf_model = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_parameters,
        n_iter=5,
        scoring="f1",
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )

    rf_model.fit(train_features, train_labels)

    joblib.dump(rf_model, "./models/berta_multiazter_rf.pkl", compress=1)

    train_output = rf_model.predict(train_features)
    test_output = rf_model.predict(test_features)

    train_score = f1_score(train_labels, train_output, average="macro")
    test_score = f1_score(test_labels, test_output, average="macro")

    print("CV best parameters: ", rf_model.best_params_)
    print("CV best results: ", rf_model.cv_results_)

    print("Training BERTA-MultiAzter score", train_score)
    print("Testing BERTA-MultiAzter score", test_score)
