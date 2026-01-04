from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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




def train_models(model_name: str, X_train, y_train, random_state=42):
    if model_name == "Random Forest":
        return train_random_forest(X_train, y_train, random_state=random_state)
    if model_name == "Logistic Regression":
        return train_logistic_regression(X_train, y_train, random_state=random_state)
    if model_name == "SVM":
        return train_svm(X_train, y_train, random_state=random_state)  
    raise ValueError(f"Unknown model: {model_name}")


    