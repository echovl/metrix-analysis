import os
import pickle

import optuna
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from transformers import AutoTokenizer, TFRobertaModel

from common import compute_evaluation_scores, merge_scores, save_autex_predictions
from dataloader import load_autextification_pucp_features
from datasets import load_dataset

MODEL_NAME = "roberta_bne_pucp_freeze"

train_dataset = load_dataset(
    "symanto/autextification2023", "detection_es", split="train"
)
test_dataset = load_dataset("symanto/autextification2023", "detection_es", split="test")

feat_train, feat_test = load_autextification_pucp_features()

x_train = np.array(train_dataset["text"])
x_test = np.array(test_dataset["text"])

y_train = np.array(train_dataset["label"])
y_test = np.array(test_dataset["label"])

x_train, x_val, y_train, y_val, feat_train, feat_val = train_test_split(
    x_train, y_train, feat_train, test_size=0.20, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")


def tokenize(texts: list[str]):
    return tokenizer(
        texts, return_tensors="tf", padding=True, max_length=128, truncation=True
    )


x_train_tokenized = tokenize(list(x_train))
x_val_tokenized = tokenize(list(x_val))
x_test_tokenized = tokenize(list(x_test))


def create_model(
    learning_rate: float,
    dense_1_size: int,
    dense_2_size: int,
    dense_3_size: int,
    dropout: float,
):
    input_ids = layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    features = layers.Input(
        shape=(182,),
        dtype=tf.float32,
        name="features",
    )

    roberta_model = TFRobertaModel.from_pretrained(
        # "PlanTL-GOB-ES/roberta-base-bne",
        "echovl/roberta-bne-autex",
        # from_pt=True,
        output_hidden_states=True,
    )
    roberta_model.trainable = False

    outputs = roberta_model(input_ids, attention_mask=attention_mask)
    cls_token = outputs.hidden_states[-1][:, 0, :]

    features_proj = layers.Dense(
        dense_1_size, activation="relu", kernel_regularizer=l2(1e-4)
    )(features)
    cls_token_proj = layers.Dense(
        dense_2_size, activation="relu", kernel_regularizer=l2(1e-4)
    )(cls_token)

    x = layers.Concatenate()([cls_token_proj, features_proj])
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_3_size, activation="relu", kernel_regularizer=l2(1e-4))(x)
    output = layers.Dense(1, activation="sigmoid", kernel_regularizer=l2(1e-4))(x)

    model = keras.Model(inputs=[input_ids, attention_mask, features], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
        loss="binary_crossentropy",
    )

    return model, roberta_model


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 3e-5)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    epochs = trial.suggest_categorical("epochs", [1])
    dense_1_size = trial.suggest_categorical("dense_1_size", [8, 16, 32])
    dense_2_size = trial.suggest_categorical("dense_2_size", [8, 16, 32, 64])
    dense_3_size = trial.suggest_categorical("dense_3_size", [8, 16, 32, 64])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    for train_index, val_index in kf.split(x_train):
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]
        feat_train_fold = feat_train[train_index]
        feat_val_fold = feat_train[val_index]

        # Convert indices to tensorflow tensors for proper indexing
        train_indices = tf.constant(train_index, dtype=tf.int32)
        val_indices = tf.constant(val_index, dtype=tf.int32)

        x_train_fold = {
            "input_ids": tf.gather(x_train_tokenized["input_ids"], train_indices),
            "attention_mask": tf.gather(
                x_train_tokenized["attention_mask"], train_indices
            ),
            "features": feat_train_fold,
        }
        x_val_fold = {
            "input_ids": tf.gather(x_train_tokenized["input_ids"], val_indices),
            "attention_mask": tf.gather(
                x_train_tokenized["attention_mask"], val_indices
            ),
            "features": feat_val_fold,
        }

        model, _ = create_model(
            learning_rate=lr,
            dense_1_size=dense_1_size,
            dense_2_size=dense_2_size,
            dense_3_size=dense_3_size,
            dropout=dropout,
        )
        history = model.fit(
            x_train_fold,
            y_train_fold,
            validation_data=(
                x_val_fold,
                y_val_fold,
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping],
        )

        _, accuracy = model.evaluate(
            x_val_fold,
            y_val_fold,
            verbose=1,
        )
        print(f"K-Fold score: {accuracy}")
        cv_scores.append(accuracy)

    return np.mean(cv_scores)


def train_model():
    # Best hyperparameters: {'lr': 2.8786545893870118e-05, 'batch_size': 16, 'epochs': 1, 'dense_1_size': 8, 'dense_2_size': 64, 'dense_3_size': 16, 'dropout': 0.3}
    steps = range(1)
    lr = (2.8786545893870118e-05,)
    batch_size = 16
    epochs = 20
    dense_1_size = 8
    dense_2_size = 64
    dense_3_size = 16
    dropout = 0.3

    best_score = 0
    best_model = None
    best_model_history = None
    scores = []
    for step in steps:
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )

        model, roberta_model = create_model(
            learning_rate=lr,
            dense_1_size=dense_1_size,
            dense_2_size=dense_2_size,
            dense_3_size=dense_3_size,
            dropout=dropout,
        )
        history = model.fit(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
                "features": feat_train,
            },
            y_train,
            validation_data=(
                {
                    "input_ids": x_val_tokenized["input_ids"],
                    "attention_mask": x_val_tokenized["attention_mask"],
                    "features": feat_val,
                },
                y_val,
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping],
        )

        train_pred = model.predict(
            {
                "input_ids": x_train_tokenized["input_ids"],
                "attention_mask": x_train_tokenized["attention_mask"],
                "features": feat_train,
            }
        )
        train_output = (train_pred > 0.5).astype(int)

        val_pred = model.predict(
            {
                "input_ids": x_val_tokenized["input_ids"],
                "attention_mask": x_val_tokenized["attention_mask"],
                "features": feat_val,
            }
        )
        val_output = (val_pred > 0.5).astype(int)

        test_pred = model.predict(
            {
                "input_ids": x_test_tokenized["input_ids"],
                "attention_mask": x_test_tokenized["attention_mask"],
                "features": feat_test,
            }
        )
        test_output = (test_pred > 0.5).astype(int)

        save_autex_predictions(test_output, MODEL_NAME)

        train_scores = compute_evaluation_scores(y_train, train_output)
        val_scores = compute_evaluation_scores(y_val, val_output)
        test_scores = compute_evaluation_scores(y_test, test_output)

        scores.append(
            merge_scores(
                [train_scores, val_scores, test_scores], ["train", "val", "test"]
            )
        )

        print(
            f"Train F1 Score: {train_scores["f1_macro"]}, Val F1 Score: {val_scores["f1_macro"]}, Test F1 Score: {test_scores["f1_macro"]}"
        )

        if val_scores["f1_macro"] > best_score:
            best_score = val_scores["f1_macro"]
            best_model = model
            best_model_history = history.history

    for score in scores:
        print(
            f"Train F1 Score: {score["train_f1_macro"]}, Val F1 Score: {score['val_f1_macro']}, Test F1 Score: {score['test_f1_macro']}"
        )
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f"./results/{MODEL_NAME}_scores.csv", index=False)
    best_model.save(f"./models/{MODEL_NAME}.h5")
    with open(f"./results/{MODEL_NAME}_history.pkl", "wb") as f:
        pickle.dump(best_model_history, f)


def optimize_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print(f"Best trial: {study.best_trial}")

    # 4. Evaluate the Best Model
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    train_model()
