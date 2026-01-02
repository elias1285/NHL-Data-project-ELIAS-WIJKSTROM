import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_accuracy(model, X_test, y_test):
    if model is None:
        return None

    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def rank_models(models, X_test, y_test):

    results = []

    for name, model in models.items():
        acc = evaluate_accuracy(model, X_test, y_test)

        if acc is not None:
            results.append({
                "model": name,
                "accuracy": acc
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="accuracy",
        ascending=False
    ).reset_index(drop=True)

    return results_df

