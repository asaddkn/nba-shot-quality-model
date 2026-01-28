import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load league data
# -----------------------------
league_df = pd.read_csv("FULL_LEAGUE_SHOTS_25_26.csv")

# Feature engineering (correct way)
league_df["angle"] = np.arctan2(league_df["LOC_Y"], league_df["LOC_X"])
league_df["distance"] = league_df["SHOT_DISTANCE"]

# -----------------------------
# Train model on league data
# -----------------------------
X = league_df[["distance", "angle"]]
y = league_df["SHOT_MADE_FLAG"]

model = LogisticRegression(max_iter=300)
model.fit(X, y)

# -----------------------------
# Score every shot with xFG
# -----------------------------
league_df["xFG"] = model.predict_proba(X)[:, 1]

league_df.to_csv("FULL_LEAGUE_SCORED_25_26.csv", index=False)

print("Saved FULL_LEAGUE_SCORED_25_26.csv with xFG")
