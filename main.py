from pathlib import Path
from src.data_loader import load_seasons

def main():
    df = load_seasons()

    df = df.rename(columns={"Round reached": "round_reached"})
    
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df = df.sort_values(by="season")

    cols = ["season"] + [c for c in df.columns if c != "season"]
    df = df[cols]

    cols = []
    for c in df.columns:
        if c not in ["made_playoffs", "round_reached"]:
            cols.append(c)
    cols.append("made_playoffs")
    cols.append("round_reached")
    df = df[cols]

    # we want to export the data to data/clean
    Path("data/clean").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/clean/all_seasons_clean.csv", index=False)
if __name__ == "__main__":
    main()
