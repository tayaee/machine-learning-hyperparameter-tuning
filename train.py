import os

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

os.makedirs("data", exist_ok=True)


def train_and_save_models():
    df = pd.read_csv("EasyVisa.csv")
    X = df.drop(["case_id", "case_status"], axis=1)
    y = df["case_status"].apply(lambda x: 1 if x == "Certified" else 0)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    base_configs = {
        # "Logistic_Regression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        "Decision_Tree": (
            DecisionTreeClassifier(random_state=1),
            {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        ),
        "Random_Forest": (RandomForestClassifier(random_state=1), {"n_estimators": [50, 100], "max_depth": [None, 10]}),
        "AdaBoost": (AdaBoostClassifier(random_state=1), {"n_estimators": [50, 100], "learning_rate": [0.1, 1.0]}),
        "Gradient_Boosting": (
            GradientBoostingClassifier(random_state=1),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        ),
        "Bagging": (BaggingClassifier(random_state=1), {"n_estimators": [10, 20]}),
        # "Extra_Trees": (ExtraTreesClassifier(random_state=1), {"n_estimators": [50, 100]}),
        "XGBoost": (
            XGBClassifier(random_state=1, eval_metric="logloss"),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        ),
        "LightGBM": (LGBMClassifier(random_state=1), {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}),
        # "SVC": (SVC(probability=True, random_state=1), {"C": [1, 10], "kernel": ["rbf", "linear"]}),
        # "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
        # "GaussianNB": (GaussianNB(), {}),
        # "DT_Pruned": (
        #     DecisionTreeClassifier(class_weight={0: 0.17, 1: 0.83}, random_state=1),
        #     {"max_depth": [3, 5, 7]},
        # ),
        # "Bagging_LR": (
        #     BaggingClassifier(base_estimator=LogisticRegression(), random_state=1),
        #     {"n_estimators": [5, 10]},
        # ),
        # "Extra_Trees_Gini": (ExtraTreesClassifier(criterion="gini", random_state=1), {"max_depth": [None, 10]}),
    }

    f1_results = {}
    for name, (model, _) in base_configs.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_results[name] = f1_score(y_test, y_pred)
        joblib.dump(model, f"data/{name}.joblib")

    top_3_names: list = sorted(f1_results, key=f1_results.get, reverse=True)[:3]
    for i, name in enumerate(top_3_names):
        print(f"Tuning {name}...")
        model_obj, params = base_configs[name]
        if params:
            tuned_search = RandomizedSearchCV(model_obj, params, n_iter=5, scoring="f1", cv=3, random_state=1)
            tuned_search.fit(X_train, y_train)
            best_model = tuned_search.best_estimator_
        else:
            best_model = model_obj.fit(X_train, y_train)

        joblib.dump(best_model, f"data/Tuned_Model_{i + 1}_{name}.joblib")

    joblib.dump(X_train.columns, "data/model_columns.joblib")
    print("Done. All models saved.")


if __name__ == "__main__":
    train_and_save_models()
