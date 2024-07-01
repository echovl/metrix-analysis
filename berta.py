import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFRobertaForSequenceClassification,
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
    # TODO: Add epochs, optimize
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


def train_berta_multiazter_model():
    train_dataset = load_dataset(
        "symanto/autextification2023", "detection_es", split="train"
    )
    # test_dataset = load_dataset(
    #     "symanto/autextification2023", "detection_es", split="test"

    tokenizer = AutoTokenizer.from_pretrained(
        "bertin-project/bertin-roberta-base-spanish"
    )
    tokenized_data = tokenizer(
        train_dataset["text"][:10], return_tensors="np", padding=True
    )

    # Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
    tokenized_data = dict(tokenized_data)

    model = TFRobertaForSequenceClassification.from_pretrained(
        "bertin-project/bertin-roberta-base-spanish"
    )

    train_output = model(**tokenized_data)

    print(train_output)
