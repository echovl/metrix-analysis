import numpy as np
import tensorflow.keras as keras
from common import get_cohmetrix_dataset_grouped, get_multiazter_dataset_grouped
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


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


if __name__ == "__main__":
    # train_berta_extended_model_keras("baseline", None, None)
    # train_grouped_metrics_model()
    # train_roberta_bne_model()
    train_grouped_merged_metrics_model()
