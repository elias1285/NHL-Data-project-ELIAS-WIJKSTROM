import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_seasons():
    all_seasons = []

    for reg_file in DATA_DIR.glob("*-regular-season.csv"):
        season = reg_file.name.replace("-regular-season.csv", "")
        playoff_file = DATA_DIR / f"{season}-playoffs.csv"

        # Load the data
        reg_df = pd.read_csv(reg_file)
        po_df = pd.read_csv(playoff_file)

        # Add season column
        reg_df["season"] = season

        # Merge the playoffs outcome into regular season data
        merged_df = reg_df.merge(
            po_df[["Team", "Round reached"]],
            on="Team",
            how="left"
        )

        
        merged_df["Round reached"] = merged_df["Round reached"].fillna(0).astype(int)
        merged_df["made_playoffs"] = (merged_df["Round reached"] > 0).astype(int)

        all_seasons.append(merged_df)

    return pd.concat(all_seasons, ignore_index=True)

