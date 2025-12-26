

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# XGBoost optional 
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        max_iter=200,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, random_state=42):
    """Train SVM model."""
    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """Train XGBoost model (if installed)."""
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

