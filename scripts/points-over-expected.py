import pandas as pd

df = pd.read_csv("shai_scored_league_xfg.csv")

# Points over expected per shot (2 points assumed for now; we refine for 3s later)
df["poe"] = (df["SHOT_MADE_FLAG"] - df["xFG"]) * 2

total_poe = df["poe"].sum()

print("Total Points Over Expected:", round(total_poe, 2))
print("Per game:", round(total_poe / 45, 2))  # ~45 games so far
