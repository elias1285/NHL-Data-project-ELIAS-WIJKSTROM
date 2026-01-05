import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Select top 16 teams
def get_top_16(playoff_model, X_season):
    probs = playoff_model.predict_proba(X_season)[:, 1]
    return np.argsort(probs)[-16:]


# Playoff accuracy
def playoff_accuracy(y_true, top_16_idx):
    y_pred = np.zeros_like(y_true)
    y_pred[top_16_idx] = 1
    return float((y_pred == y_true).mean())


# Rank playoff teams
def rank_teams(round_model, X_16):
    probs = round_model.predict_proba(X_16)
    classes = list(round_model.classes_)

    scores = np.zeros(len(X_16))
    for j, c in enumerate(classes):
        scores += c * probs[:, j]

    return scores


# Assign playoff rounds
def assign_round_reached(df_ranked):
    rounds = np.ones(16, dtype=int)
    rounds[0] = 5
    rounds[1] = 4
    rounds[2:4] = 3
    rounds[4:8] = 2
    rounds[8:16] = 1

    out = df_ranked.copy().reset_index(drop=True)
    out["pred_round_reached"] = rounds
    return out


# Evaluate one season
def evaluate_season(playoff_model, round_model, df_season, X_season):
    y_playoffs = df_season["made_playoffs"].to_numpy()
    y_rounds = df_season["round_reached"].to_numpy()

    top16 = get_top_16(playoff_model, X_season)
    p_acc = playoff_accuracy(y_playoffs, top16)

    df_16 = df_season.iloc[top16].reset_index(drop=True)
    X_16 = X_season.iloc[top16].reset_index(drop=True)

    scores = rank_teams(round_model, X_16)
    order = np.argsort(scores)[::-1]

    df_ranked = df_16.iloc[order].reset_index(drop=True)
    pred_df = assign_round_reached(df_ranked[["season", "team"]])

    true_rounds = y_rounds[top16][order]
    r_acc = float(
        (pred_df["pred_round_reached"].to_numpy() == true_rounds).mean()
    )

    playoff_teams = (
        df_season.iloc[top16][["season", "team"]]
        .sort_values("team")
        .reset_index(drop=True)
    )

    return p_acc, r_acc, playoff_teams, pred_df


# Evaluate all models
def evaluate_models(models, round_models, df, X, test_seasons):
    playoff_acc = pd.DataFrame(index=models.keys(), columns=test_seasons)
    round_acc = pd.DataFrame(index=models.keys(), columns=test_seasons)

    pred_playoffs = {m: {} for m in models}
    pred_rounds = {m: {} for m in models}

    for name in models:
        playoff_model = models[name]
        round_model = round_models[name]

        for season in test_seasons:
            mask = df["season"] == season
            df_season = df.loc[mask].reset_index(drop=True)
            X_season = X.loc[mask].reset_index(drop=True)

            p_acc, r_acc, p_teams, r_pred = evaluate_season(
                playoff_model, round_model, df_season, X_season
            )

            playoff_acc.loc[name, season] = p_acc
            round_acc.loc[name, season] = r_acc

            pred_playoffs[name][season] = p_teams
            pred_rounds[name][season] = r_pred

    return playoff_acc, round_acc, pred_playoffs, pred_rounds


# Collect round predictions
def collect_round_predictions(models, round_models, df, X, test_seasons):
    collected = {}

    for name in models:
        y_true_all = []
        y_pred_all = []

        for season in test_seasons:
            mask = df["season"] == season
            df_season = df.loc[mask].reset_index(drop=True)
            X_season = X.loc[mask].reset_index(drop=True)

            _, _, _, pred_df = evaluate_season(
                models[name], round_models[name], df_season, X_season
            )

            y_rounds = df_season["round_reached"].to_numpy()
            top16 = get_top_16(models[name], X_season)

            X_16 = X_season.iloc[top16].reset_index(drop=True)

            scores = rank_teams(round_models[name], X_16)
            order = np.argsort(scores)[::-1]

            y_true_all.extend(y_rounds[top16][order])
            y_pred_all.extend(pred_df["pred_round_reached"].to_numpy())

        collected[name] = (np.array(y_true_all), np.array(y_pred_all))

    return collected


# Confusion matrix
def confusion_matrix_round_reached(y_true, y_pred):
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"correct:{l}" for l in labels],
        columns=[f"predicted:{l}" for l in labels],
    )


# Per round accuracy
def per_round_accuracy(y_true, y_pred):
    results = {}
    for r in [1, 2, 3, 4, 5]:
        idx = y_true == r
        results[r] = float((y_pred[idx] == r).mean()) if idx.sum() > 0 else None
    return results


# Format output
def format_table(df, decimals=3):
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(
            lambda v: f"{v:.{decimals}f}" if pd.notna(v) else "NA"
        )
    return out