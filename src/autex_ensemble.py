import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from dataloader import load_autextification_dataset, load_autextification_pucp_features


def evaluate_xgboost():
    _, train_labels, _, test_labels = load_autextification_dataset()
    train_pucpmetrix_features, test_pucpmetrix_features = (
        load_autextification_pucp_features()
    )

    print("PUCPMetrix features shape:", train_pucpmetrix_features.shape)
    print("PUCPMetrix test features shape:", test_pucpmetrix_features.shape)

    xgboost_model = XGBClassifier()
    xgboost_model.load_model("xgboost_model.json")

    train_predicted = xgboost_model.predict(train_pucpmetrix_features)
    test_predicted = xgboost_model.predict(test_pucpmetrix_features)

    train_score = f1_score(train_labels, train_predicted, average="macro")
    test_score = f1_score(test_labels, test_predicted, average="macro")

    print(f"Train score: {train_score}, Test score: {test_score}")


if __name__ == "__main__":
    evaluate_xgboost()
