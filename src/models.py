from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
    max_iter=200,
    random_state=random_state,
    class_weight="balanced"
    )


    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, random_state=42):
    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    if not HAS_XGB:
        return None

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model

