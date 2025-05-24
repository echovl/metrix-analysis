import os

import optuna

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFRobertaModel

from datasets import load_dataset

train_dataset = load_dataset(
    "symanto/autextification2023", "detection_es", split="train"
)
test_dataset = load_dataset("symanto/autextification2023", "detection_es", split="test")

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


def create_model(learning_rate: float, dense_size: int, dropout: float):
    input_ids = layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    roberta_model = TFRobertaModel.from_pretrained(
        "PlanTL-GOB-ES/roberta-base-bne",
        from_pt=True,
        output_hidden_states=True,
    )

    outputs = roberta_model(input_ids, attention_mask=attention_mask)
    cls_output = outputs.hidden_states[-1][:, 0, :]

    x = layers.Dropout(dropout)(cls_output)
    x = layers.Dense(dense_size, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
        loss="binary_crossentropy",
    )

    return model


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 3e-5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [1, 2, 3])
    dense_size = trial.suggest_categorical("dense_size", [256, 512, 768])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    for train_index, val_index in kf.split(x_train):
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]

        # Convert indices to tensorflow tensors for proper indexing
        train_indices = tf.constant(train_index, dtype=tf.int32)
        val_indices = tf.constant(val_index, dtype=tf.int32)

        x_train_fold = {
            "input_ids": tf.gather(x_train_tokenized["input_ids"], train_indices),
            "attention_mask": tf.gather(
                x_train_tokenized["attention_mask"], train_indices
            ),
        }
        x_val_fold = {
            "input_ids": tf.gather(x_train_tokenized["input_ids"], val_indices),
            "attention_mask": tf.gather(
                x_train_tokenized["attention_mask"], val_indices
            ),
        }

        model = create_model(learning_rate=lr, dense_size=dense_size, dropout=dropout)
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


def train_roberta_model():
    steps = range(5)
    for step in steps:
        best_lr = 1e-5
        best_batch_size = 4
        best_epochs = 1
        best_dense_size = 128
        best_dropout = 0.1

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True
        )

        model = create_model(
            learning_rate=best_lr, dense_size=best_dense_size, dropout=best_dropout
        )
        history = model.fit(
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
            epochs=best_epochs,
            batch_size=best_batch_size,
            verbose=1,
            callbacks=[early_stopping],
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

        print(
            f"Train F1 Score: {train_score}, Val F1 Score: {val_score}, Test F1 Score: {test_score}"
        )


def optimize_roberta_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print(f"Best trial: {study.best_trial}")

    # 4. Evaluate the Best Model
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    optimize_roberta_model()
