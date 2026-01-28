import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Rebuild model from league data
# -----------------------------
league_df = pd.read_csv("FULL_LEAGUE_SHOTS_25_26.csv")

league_df["angle"] = np.arctan2(league_df["LOC_Y"], league_df["LOC_X"])
league_df["distance"] = np.sqrt(league_df["LOC_X"]**2 + league_df["LOC_Y"]**2)

X = league_df[["distance", "angle"]]
y = league_df["SHOT_MADE_FLAG"]

model = LogisticRegression(max_iter=300)
model.fit(X, y)

# -----------------------------
# Load Shai shots
# -----------------------------
shai_df = pd.read_csv("shai_shots_25_26.csv")

shai_df["angle"] = np.arctan2(shai_df["LOC_Y"], shai_df["LOC_X"])
shai_df["distance"] = np.sqrt(shai_df["LOC_X"]**2 + shai_df["LOC_Y"]**2)

# Score xFG
shai_df["xFG"] = model.predict_proba(shai_df[["distance", "angle"]])[:, 1]

shai_df.to_csv("shai_scored_full_league_xfg.csv", index=False)

print("Saved Shai with new xFG")
