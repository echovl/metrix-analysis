import json
import pathlib
import subprocess
import tempfile
from collections import OrderedDict
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MULTIAZTER_PYTHON_PATH = "/home/echo/dev/metrix-box/MultiAzterTest/.venv/bin/python"

LABEL_HUMAN = 0
LABEL_GENERATED = 1


def multiazter_metrics(texts: [str], language: str = "spanish") -> [Dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, text in enumerate(texts):
            with open(f"{tmpdir}/{i}.txt", "w") as f:
                f.write(text)

        texts_path = [f"{tmpdir}/{i}.txt" for i, _ in enumerate(texts)]

        print(texts_path)

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
            stderr=subprocess.STDOUT,
            # capture_output=True,
            # text=True,
        )

        raw_result = p.stdout.split("__MULTIAZTER_RESULT__")[1]
        metrics = json.loads(raw_result)

        return metrics


# text = "Multiaztertest is an upgrade to the Aztertest application meant to evaluate texts in various languages by calculating multiple metrics and indicators of the texts' content and analyzing those results to determine the complexity level of those texts."
# metrics = multiazter_metrics(text, language="spanish")

train_dataset = load_dataset(
    "symanto/autextification2023", "detection_es", split="train"
)
test_dataset = load_dataset("symanto/autextification2023", "detection_es", split="test")

print(train_dataset)
print(train_dataset[0])

x_train = np.array([])
y_train = np.array([])

texts = []
for data in train_dataset:
    texts.append(data["text"])
    # metrics = multiazter_metrics(data["text"], language="spanish")
    # print("items", OrderedDict(metrics).items())

metrics = multiazter_metrics(texts, language="spanish")

print(metrics)

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
