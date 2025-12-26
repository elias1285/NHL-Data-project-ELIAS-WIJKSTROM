def print_top_features(model, feature_names, model_name, top_n=10):
    """
    Print top features for tree-based or linear models.
    """
    print("\n" + "-" * 50)
    print(f"Top {top_n} features for {model_name}")
    print("-" * 50)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])

    else:
        print("This model does not provide feature importances.")
        return

    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, val in feature_importance[:top_n]:
        print(f"{feat}: {val:.4f}")

