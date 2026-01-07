import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_clean_data
from src.models import train_models
from src.evaluation import (
    evaluate_models,
    collect_playoff_predictions,
    confusion_matrix_playoffs,
    collect_round_predictions,
    confusion_matrix_round_reached,
    per_round_accuracy,
    format_table,
)


def make_X(df):
    remove_cols = ["team", "season", "made_playoffs", "round_reached"]
    return df.drop(columns=[c for c in remove_cols if c in df.columns], errors="ignore")

# get top 10 random forest features
def random_forest_important_features(model, feature_names, top_n=10):
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )

# get top 10 coefficient (absolut value)
def logistic_regression_coefficients(model, feature_names, top_n=10):
    return (
        pd.Series(model.coef_[0], index=feature_names)
        .sort_values(key=abs, ascending=False)
        .head(top_n)
    )

# scale the data
def scale_training_data(X_train, X_all):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_all_scaled = pd.DataFrame(
        scaler.transform(X_all),
        columns=X_all.columns,
        index=X_all.index,
    )

    return X_train_scaled, X_all_scaled, scaler


def main():
    
    df = load_clean_data(save_csv=False)

    # features and targets
    X_all = make_X(df)
    y_playoffs = df["made_playoffs"]

    # season lists
    all_seasons = sorted(df["season"].unique())
    training_seasons = all_seasons[:6]
    testing_seasons = all_seasons[-4:]

    print(f"TRAIN SEASONS: {training_seasons}")
    print(f"TEST  SEASONS: {testing_seasons}")

    # playoff training split
    training_rows = df["season"].isin(training_seasons)
    X_train_playoffs = X_all.loc[training_rows]
    y_train_playoffs = y_playoffs.loc[training_rows]

    # scale using playoff training data fit only on training data
    X_train_playoffs_scaled, X_all_scaled, scaler = scale_training_data(
        X_train_playoffs, X_all
    )

    # round training split
    playoff_teams_df = df[df["made_playoffs"] == 1]
    playoff_training_rows = playoff_teams_df["season"].isin(training_seasons)

    X_train_rounds = make_X(playoff_teams_df).loc[playoff_training_rows]
    y_train_rounds = playoff_teams_df.loc[playoff_training_rows, "round_reached"]

    # make sure columns match the scaler fit columns
    X_train_rounds = X_train_rounds[X_train_playoffs.columns]

    # apply same scaler to round features
    X_train_rounds_scaled = pd.DataFrame(
        scaler.transform(X_train_rounds),
        columns=X_train_rounds.columns,
        index=X_train_rounds.index,
    )

    # model names
    model_names = ["Random Forest", "Logistic Regression", "SVM"]

    playoff_models = {}
    round_models = {}

    # train models on scaled data
    for model_name in model_names:
        playoff_models[model_name] = train_models(
            model_name, X_train_playoffs_scaled, y_train_playoffs
        )
        round_models[model_name] = train_models(
            model_name, X_train_rounds_scaled, y_train_rounds
        )

    # model results
    playoff_accuracy, round_accuracy, predicted_playoffs, predicted_rounds = evaluate_models(
        playoff_models,
        round_models,
        df,
        X_all_scaled,
        testing_seasons,
    )

    #Trminal output
    for model_name in model_names:
        print("\n" + "-" * 90)
        print(f"MODEL: {model_name}")
        print("-" * 90)

        for season in testing_seasons:
            p_acc = float(playoff_accuracy.loc[model_name, season])
            r_acc = float(round_accuracy.loc[model_name, season])

            print("\n" + "." * 80)
            print(f"SEASON: {season}")
            print(f"Playoff accuracy: {p_acc:.3f}")
            print(f"Round accuracy:   {r_acc:.3f}")
            print("." * 80)

            print("\nPlayoff teams:")
            print(predicted_playoffs[model_name][season].to_string(index=False))

            print("\nRounds reached:")
            print(predicted_rounds[model_name][season].to_string(index=False))

    # playoff confusion matrices
    playoff_results = collect_playoff_predictions(
        playoff_models, df, X_all_scaled, testing_seasons
    )

    print("\n" + "=" * 90)
    print("PLAYOFF CONFUSION MATRICES")
    for model_name, (y_true, y_pred) in playoff_results.items():
        print("\n" + "-" * 60)
        print(f"MODEL: {model_name}")
        print(confusion_matrix_playoffs(y_true, y_pred))

    # round confusion matrices
    round_results = collect_round_predictions(
        playoff_models, round_models, df, X_all_scaled, testing_seasons
    )

    print("\n" + "=" * 90)
    print("ROUND CONFUSION MATRICES")
    for model_name, (y_true, y_pred) in round_results.items():
        print("\n" + "-" * 60)
        print(f"MODEL: {model_name}")
        print(confusion_matrix_round_reached(y_true, y_pred))
        print("Per-round accuracy:", per_round_accuracy(y_true, y_pred))

    # summary tables
    print("\nPLAYOFF ACCURACY")
    print(format_table(playoff_accuracy))

    print("\nROUND ACCURACY")
    print(format_table(round_accuracy))

    # save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # save accuracy tables
    format_table(playoff_accuracy).to_csv(results_dir / "playoff_accuracy.csv", index=True)
    format_table(round_accuracy).to_csv(results_dir / "round_accuracy.csv", index=True)

    # save confusion matrices
    for model_name, (y_true, y_pred) in playoff_results.items():
        cm_df = confusion_matrix_playoffs(y_true, y_pred)
        cm_df.to_csv(results_dir / f"playoff_confusion_matrix_{model_name}.csv", index=True)

    for model_name, (y_true, y_pred) in round_results.items():
        cm_df = confusion_matrix_round_reached(y_true, y_pred)
        cm_df.to_csv(results_dir / f"round_confusion_matrix_{model_name}.csv", index=True)

    # save per-round accuracy
    per_round_rows = []
    for model_name, (y_true, y_pred) in round_results.items():
        acc = per_round_accuracy(y_true, y_pred)
        row = {"model": model_name}
        row.update({f"round_{k}": v for k, v in acc.items()})
        per_round_rows.append(row)

    pd.DataFrame(per_round_rows).to_csv(results_dir / "per_round_accuracy.csv", index=False)

    # feature and coefficient importance

    rf_playoff_top10 = random_forest_important_features(
        playoff_models["Random Forest"],
        X_train_playoffs_scaled.columns,
        top_n=10,
    )

    rf_rounds_top10 = random_forest_important_features(
        round_models["Random Forest"],
        X_train_rounds_scaled.columns,
        top_n=10,
    )

    rf_playoff_top10.to_csv(results_dir / "random_forest_top10_features_playoffs.csv")
    rf_rounds_top10.to_csv(results_dir / "random_forest_top10_features_rounds.csv")

     

    lr_playoff_top10 = logistic_regression_coefficients(
        playoff_models["Logistic Regression"],
        X_train_playoffs_scaled.columns,
        top_n=10,
    )

    lr_rounds_top10 = logistic_regression_coefficients(
        round_models["Logistic Regression"],
        X_train_rounds_scaled.columns,
        top_n=10,
    )

    lr_playoff_top10.to_csv(results_dir / "logistic_regression_top10_coefficients_playoffs.csv")
    lr_rounds_top10.to_csv(results_dir / "logistic_regression_top10_coefficients_rounds.csv")


if __name__ == "__main__":
    main()
