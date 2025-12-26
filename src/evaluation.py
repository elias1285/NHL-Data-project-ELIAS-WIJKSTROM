from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("-" * 50)
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return acc

