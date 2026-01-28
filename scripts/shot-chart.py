import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("shai_shots_25_26.csv")

made = df[df["SHOT_MADE_FLAG"] == 1]
missed = df[df["SHOT_MADE_FLAG"] == 0]

plt.figure(figsize=(6,5))
plt.scatter(missed["LOC_X"], missed["LOC_Y"], alpha=0.3, label="Missed")
plt.scatter(made["LOC_X"], made["LOC_Y"], alpha=0.6, label="Made")

plt.title("Shai Shot Map 2025-26")
plt.legend()
plt.xlim(-250, 250)
plt.ylim(-50, 450)
plt.show()
