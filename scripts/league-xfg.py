import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# -----------------------------
# Load league dataset
# -----------------------------
league_df = pd.read_csv("FULL_LEAGUE_SHOTS_25_26.csv")

# -----------------------------
# Feature engineering
# -----------------------------
league_df["angle"] = np.arctan2(league_df["LOC_Y"], league_df["LOC_X"])
league_df["distance"] = np.sqrt(league_df["LOC_X"]**2 + league_df["LOC_Y"]**2)

# -----------------------------
# Train xFG model
# -----------------------------
X = league_df[["distance", "angle"]]
y = league_df["SHOT_MADE_FLAG"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print("League rows:", len(league_df))
print("League make rate:", y.mean())
print("ROC AUC:", auc)

# -----------------------------
# Save model-ready league data (optional)
# -----------------------------
league_df.to_csv("league_with_features.csv", index=False)
