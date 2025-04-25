import numpy as np
import pandas as pd
import seaborn


def load_dataset(dataset_name: str):
    if dataset_name == "titanic":
        dataset = seaborn.load_dataset("titanic")

        # Replace 'survived' with 'target'
        dataset["target"] = dataset["survived"]
        dataset = dataset.drop(columns=["survived"])

        # Drop irrelevant or redundant columns
        dataset = dataset.drop(columns=[
            "deck", "embark_town", "alive", "class", "who", "adult_male"
        ], errors="ignore")

        # Fill missing values
        dataset["age"] = dataset["age"].fillna(dataset["age"].median())
        dataset["embarked"] = dataset["embarked"].fillna(dataset["embarked"].mode()[0])

        # Encode categorical variables
        dataset = pd.get_dummies(dataset, columns=["sex", "embarked"], drop_first=True)

        # Ensure target is integer
        dataset["target"] = dataset["target"].astype(int)

        # Split into features and target
        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        return X, y
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")
