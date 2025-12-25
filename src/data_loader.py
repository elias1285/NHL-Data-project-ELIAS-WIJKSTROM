import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_seasons():
    all_seasons = []

    for reg_file in DATA_DIR.glob("*-regular-season.csv"):
        season = reg_file.name.replace("-regular-season.csv", "")
        playoff_file = DATA_DIR / f"{season}-playoffs.csv"

        reg_df = pd.read_csv(reg_file)
        po_df = pd.read_csv(playoff_file)

        reg_df["season"] = season

        merged_df = reg_df.merge(
            po_df[["Team", "Round reached"]],
            on="Team",
            how="left"
        )

        merged_df["Round reached"] = merged_df["Round reached"].fillna(0).astype(int)
        merged_df["made_playoffs"] = (merged_df["Round reached"] > 0).astype(int)

        all_seasons.append(merged_df)

    return pd.concat(all_seasons, ignore_index=True)


def clean_data(df):
    df = df.rename(columns={"Round reached": "round_reached"})

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df = df.sort_values(by="season").reset_index(drop=True)

    standings_vars = ["W", "L", "OTL", "ROW", "Points", "Point %", "Point%"]
    df = df.drop(columns=[c for c in standings_vars if c in df.columns], errors="ignore")

    cols = ["season"] + [c for c in df.columns if c != "season"]
    df = df[cols]

    target_cols = ["made_playoffs", "round_reached"]
    other_cols = [c for c in df.columns if c not in target_cols]
    df = df[other_cols + target_cols]

    return df


def load_clean_data(save_csv=True):
    df = load_seasons()
    df = clean_data(df)

    if save_csv:
        Path("data/clean").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/clean/all_seasons_clean.csv", index=False)

    return df

