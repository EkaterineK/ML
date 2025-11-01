# for model training and tuning

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_all_models(X_train, y_train):
    """
    Trains Logistic Regression, Random Forest, and SVM models
    using GridSearchCV hyperparameter tuning.
    Returns dictionary of best estimators.
    """

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }


    params = {
        "LogisticRegression": {"C": [0.1, 1, 10]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 7, 9]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    }

    best_models = {}
    for name, model in models.items():
        print(f"Tuning {name} ...")
        grid = GridSearchCV(model, params[name], cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"{name} best params: {grid.best_params_}\n")

    print("All models trained and optimized successfully.")
    return best_models
