import pandas as pd

df = pd.read_csv("shai_scored_full_league_xfg.csv")

# Create distance zones
bins = [0, 4, 10, 16, 23, 40]
labels = ["Rim", "Short Mid", "Midrange", "Long Mid", "3PT"]

df["zone"] = pd.cut(df["distance"], bins=bins, labels=labels)

summary = df.groupby("zone").agg(
    shots=("SHOT_MADE_FLAG", "count"),
    actual_fg=("SHOT_MADE_FLAG", "mean"),
    expected_fg=("xFG", "mean")
).reset_index()

summary["difference"] = summary["actual_fg"] - summary["expected_fg"]

print(summary)
