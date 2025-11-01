# for data loading and cleaning

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(path="../data/diabetes.csv"):
    """
    Loads the diabetes dataset, cleans missing values, scales features,
    and splits into train/test sets.
    Returns: X_train, X_test, y_train, y_test
    """

    df = pd.read_csv(path)

    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
