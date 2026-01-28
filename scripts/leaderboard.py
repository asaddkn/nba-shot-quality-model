import pandas as pd
import numpy as np

df = pd.read_csv("FULL_LEAGUE_SCORED_25_26.csv")  # <-- your scored league file with xFG

# Ensure these exist:
# PLAYER_NAME (or player_name), SHOT_MADE_FLAG, xFG, SHOT_DISTANCE, SHOT_TYPE

name_col = "PLAYER_NAME" if "PLAYER_NAME" in df.columns else "player_name"

# PoE: points over expected (2 for 2PT, 3 for 3PT)
is_3 = df["SHOT_TYPE"].astype(str).str.contains("3PT", case=False, na=False)
points = np.where(is_3, 3.0, 2.0)

df["poe"] = (df["SHOT_MADE_FLAG"] - df["xFG"]) * points

leader = df.groupby(name_col).agg(
    shots=("SHOT_MADE_FLAG", "count"),
    fg=("SHOT_MADE_FLAG", "mean"),
    xfg=("xFG", "mean"),
    poe=("poe", "sum"),
).reset_index()

leader["poe_per_100"] = 100 * leader["poe"] / leader["shots"]
leader = leader[leader["shots"] >= 200].sort_values("poe", ascending=False)

print(leader.head(25))
leader.to_csv("leaderboard_poe.csv", index=False)
# Zones using SHOT_DISTANCE in feet
bins = [0, 4, 10, 16, 23, 40]
labels = ["Rim", "Short Mid", "Midrange", "Long Mid", "3PT+"]

df["zone"] = pd.cut(df["SHOT_DISTANCE"], bins=bins, labels=labels, include_lowest=True)

zone_leader = df.groupby([name_col, "zone"]).agg(
    shots=("SHOT_MADE_FLAG", "count"),
    poe=("poe", "sum")
).reset_index()

zone_leader["poe_per_100"] = 100 * zone_leader["poe"] / zone_leader["shots"]

zone_leader = zone_leader[zone_leader["shots"] >= 50].sort_values(["zone", "poe_per_100"], ascending=[True, False])
zone_leader.to_csv("leaderboard_zone_poe.csv", index=False)
