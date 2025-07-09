import pandas as pd

roberta_pucp_finetuned = pd.read_csv("./results/roberta_bne_pucp_scores.csv")
roberta_pucp_freezed = pd.read_csv("./results/roberta_bne_pucp_freeze_scores.csv")
roberta_pucp_stacking = pd.read_csv("./results/autextification_ensemble_stacking.csv")
roberta_pucp_voting = pd.read_csv("./results/autextification_ensemble_voting.csv")
roberta_pucp_voting_no_roberta = pd.read_csv("./results/autextification_ensemble_voting_no_roberta.csv")
roberta_pucp_xgboost = pd.read_csv("./results/autextification_roberta_proba_pucp_xgb.csv")

models = [
    ("roberta_pucp_finetuned", roberta_pucp_finetuned),
    ("roberta_pucp_freezed", roberta_pucp_freezed),
    ("roberta_pucp_stacking", roberta_pucp_stacking),
    ("roberta_pucp_voting", roberta_pucp_voting),
    ("roberta_pucp_voting_no_roberta", roberta_pucp_voting_no_roberta),
    ("roberta_pucp_xgboost", roberta_pucp_xgboost)
]

# Columns of interest
columns_of_interest = [
    "train_f1_macro", 
    "val_f1_macro", 
    "test_f1_macro", 
    "val_gen_recall", 
    "val_gen_precision", 
    "test_gen_recall", 
    "test_gen_precision"
]

for model_name, model in models:
    # Find the median value of val_f1_macro
    median_val_f1_macro = model["val_f1_macro"].median()
    
    # Find the row closest to the median val_f1_macro
    closest_idx = (model["val_f1_macro"] - median_val_f1_macro).abs().idxmin()
    median_row = model.loc[closest_idx]
    
    print(f"Model: {model_name}")
    print(f"Median val_f1_macro: {median_val_f1_macro}")
    print("Row with median val_f1_macro:")
    
    # Print only the columns of interest
    for col in columns_of_interest:
        print(f"  {col}: {median_row[col]}")
    print()
