import numpy as np
import pandas as pd

from src.data_loader import load_clean_data
from src.models import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_xgboost,
)

def target_data_to_train(target="playoffs", save_csv=False):
    df = load_clean_data(save_csv=save_csv)
    drop_from_x = ["team", "made_playoffs", "round_reached", "season"]
    

    if target == "playoffs":
        X= df.drop(columns=[c for c in drop_from_x if c in df.columns], errors="ignore")
        y = df["made_playoffs"].astype(int)
        return X, y

    if target =="round_reached":
        df_po = df[df["made_playoffs"] == 1].copy()
        X_po = df_po.drop(columns=[c for c in drop_from_x if c in df_po.columns], errors="ignore")
        y_po = df_po["round_reached"].astype(int)
        return X_po, y_po

    raise ValueError("target must be 'playoffs' or 'round_reached'")

def train_one(model_name: str, X_train, y_train, random_state=42):
    if model_name == "Random Forest":
        return train_random_forest(X_train, y_train, random_state=random_state)
    if model_name == "Logistic Regression":
        return train_logistic_regression(X_train, y_train, random_state=random_state)
    if model_name == "SVM":
        return train_svm(X_train, y_train, random_state=random_state)
    if model_name == "XGBoost":
        return train_xgboost(X_train, y_train, random_state=random_state)  
    raise ValueError(f"Unknown model: {model_name}")

def main():
    df = load_clean_data(save_csv=False)
    X, y = target_data_to_train(target="playoffs", save_csv=False)

    assert len(df) == len(X) == len(y), "df, X, y are not aligned"

    seasons = sorted(df["season"].unique())
    train_seasons = seasons[:6]
    test_seasons = seasons[-4:]

    train_mask = df["season"].isin(train_seasons)
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    model_list = ["Random Forest", "Logistic Regression", "SVM", "XGBoost"]

    # outputs 
    acc_table = pd.DataFrame(index=model_list, columns=test_seasons, dtype=float)

    print("\n" + "=" * 90)
    print(f"TRAIN SEASONS: {train_seasons}")
    print(f"TEST  SEASONS: {test_seasons}")
    print("=" * 90)

    for model_name in model_list:
        print("\n" + "-" * 90)
        print(f"MODEL: {model_name}")
        print("-" * 90)

        model = train_one(model_name, X_train, y_train, random_state=42)
        if model is None:
            print("Skipped (model not available).")
            continue

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                f"{model_name} has no predict_proba(). "
                f"(For SVM, set probability=True in train_svm.)"
            )

        for season in test_seasons:
            season_mask = df["season"] == season

            X_season = X.loc[season_mask]
            y_true = y.loc[season_mask].to_numpy()

            season_info = df.loc[season_mask, ["season", "team"]].copy().reset_index(drop=True)

            probs = model.predict_proba(X_season)[:, 1]

            # we want 16 teams
            top16_idx =np.argsort(probs)[-16:]
            y_pred = np.zeros_like(y_true)
            y_pred[top16_idx]= 1

            accuracy = (y_pred == y_true).mean()
            acc_table.loc[model_name, season] = accuracy

            predicted_teams = (
                season_info.iloc[top16_idx][["season", "team"]]
                .sort_values("team")
                .reset_index(drop=True)
            )

            print("\n" + "." * 80)
            print(f"MODEL: {model_name} | SEASON: {season}")
            print(f"Playoff prediction accuracy: {accuracy:.3f}")
            print("." * 80)
            print(predicted_teams.to_string(index=False))

    #summary table
    print("\n" + "=" * 90)
    print("ACCURACY SUMMARY (rows=models, columns=test seasons)")
    print("=" * 90)

    
    summary = acc_table.copy()
    for c in summary.columns:
        summary[c] = summary[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "NA")

    print(summary.to_string())

if __name__ == "__main__":
    main()

