#for model evaluation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
)

def evaluate_models(models, X_test, y_test):
    """
    Evaluates multiple trained models using standard classification metrics
    and saves visualizations to /results.
    Returns a DataFrame of performance metrics.
    """
    results = []

    for name, model in models.items():
        print(f"Evaluating {name} ...")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append([name, acc, prec, rec, f1, auc])

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"../results/cm_{name}.png")
        plt.close()

        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(f"../results/roc_{name}.png")
        plt.close()

        print(f"{name} done!\n")

    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
    df.to_csv("../results/model_results.csv", index=False)
    print("Results saved to /results/model_results.csv")
    return df
