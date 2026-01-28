import time
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players

SEASON = "2025-26"
SEASON_TYPE = "Regular Season"

all_players = players.get_active_players()

frames = []

for i, p in enumerate(all_players):
    name = p["full_name"]
    pid = p["id"]

    try:
        shots = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=pid,
            season_type_all_star=SEASON_TYPE,
            season_nullable=SEASON,
            context_measure_simple='FGA'
        )

        df = shots.get_data_frames()[0]

        if len(df) > 50:  # ignore tiny samples
            df["player_name"] = name
            frames.append(df)
            print(f"{i+1}/{len(all_players)} - {name}: {len(df)} shots")

        time.sleep(0.6)

    except Exception as e:
        print(f"Skipped {name}")
        time.sleep(0.6)
        continue

league_df = pd.concat(frames, ignore_index=True)
league_df.to_csv("FULL_LEAGUE_SHOTS_25_26.csv", index=False)

print("Saved full league dataset:", len(league_df))

# ---- Train xFG model on "league" ----
league_df["angle"] = np.arctan2(league_df["LOC_Y"], league_df["LOC_X"])
league_df["distance"] = np.sqrt(league_df["LOC_X"]**2 + league_df["LOC_Y"]**2)
X = league_df[["distance", "angle"]]
y = league_df["SHOT_MADE_FLAG"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print("\n=== League model diagnostics ===")
print("League rows:", len(league_df))
print("League make rate:", float(league_df['SHOT_MADE_FLAG'].mean()))
print("ROC AUC:", float(auc))

# ---- Pull Shai shots & score them ----
shai_df = fetch_player_shots(shai_id, team_id=1610612760, season=SEASON)  # OKC
shai_df = engineer_features(shai_df)
shai_df["xFG"] = model.predict_proba(shai_df[["distance", "angle"]])[:, 1]

print("\n=== Shai diagnostics ===")
print("Shai rows:", len(shai_df))
print("Shai make rate:", float(shai_df["SHOT_MADE_FLAG"].mean()))

# ---- Zone analysis ----
summary = zone_summary(shai_df)

print("\n=== Shai: Actual vs Expected by zone ===")
print(summary)

# Save outputs
league_df.to_csv("league_shots_train.csv", index=False)
shai_df.to_csv("shai_scored_league_xfg.csv", index=False)
summary.to_csv("shai_zone_summary.csv", index=False)

print("\nSaved:")
print("- league_shots_train.csv")
print("- shai_scored_league_xfg.csv")
print("- shai_zone_summary.csv")
