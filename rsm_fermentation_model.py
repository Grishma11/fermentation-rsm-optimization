"""
Response Surface Methodology for Fermentation Optimisation

This project demonstrates how Response Surface Methodology (RSM) can be used
to model lactic acid production in a simulated fermentation system.

Important:
This project uses synthetic fermentation data for educational and portfolio purposes.
It does not represent real experimental, industrial, or confidential data.
"""

# ----------------------------
# Import libraries
# ----------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ----------------------------
# STEP 1: Create output folder for figures
# ----------------------------

os.makedirs("figures", exist_ok=True)


# ----------------------------
# STEP 2: Create synthetic fermentation design table
# ----------------------------

df = pd.DataFrame({
    "Run": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Inoculum_pct": [0.1, 1.0, 10.0, 0.1, 1.0, 10.0, 0.1, 1.0, 10.0, 1.0, 1.0],
    "Time_h": [24, 24, 24, 30, 30, 30, 48, 48, 48, 30, 30]
})

# Synthetic lactic acid values created for demonstration purposes.
# These values do not represent real experimental or industrial data.
df["Lactic_Acid_gL"] = [1.8, 2.4, 2.9, 2.1, 3.0, 3.8, 2.6, 3.7, 4.5, 3.1, 3.0]


# ----------------------------
# STEP 3: Code independent variables for RSM
# ----------------------------

# A = coded inoculum variable
# 0.1% inoculum -> -1
# 1.0% inoculum -> 0
# 10.0% inoculum -> +1
df["A"] = np.log10(df["Inoculum_pct"])

# B = coded time variable
# Center point = 30 h
# 24 h -> -0.333
# 30 h -> 0
# 48 h -> +1
df["B"] = (df["Time_h"] - 30) / 18


print("\nExperimental design table:")
print(df)


# ----------------------------
# STEP 4: Fit quadratic RSM model
# ----------------------------

model = smf.ols(
    "Lactic_Acid_gL ~ A + B + A:B + I(A**2) + I(B**2)",
    data=df
).fit()

print("\nMODEL SUMMARY")
print(model.summary())


# ----------------------------
# STEP 5: Predict values for existing runs
# ----------------------------

df["Predicted_Lactic_Acid"] = model.predict(df)

print("\nObserved vs Predicted:")
print(df[[
    "Run",
    "Inoculum_pct",
    "Time_h",
    "Lactic_Acid_gL",
    "Predicted_Lactic_Acid"
]])


# ----------------------------
# STEP 6: Calculate model performance
# ----------------------------

r2 = r2_score(df["Lactic_Acid_gL"], df["Predicted_Lactic_Acid"])
rmse = np.sqrt(mean_squared_error(df["Lactic_Acid_gL"], df["Predicted_Lactic_Acid"]))
mae = mean_absolute_error(df["Lactic_Acid_gL"], df["Predicted_Lactic_Acid"])

print("\nMODEL PERFORMANCE")
print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.3f} g/L")
print(f"MAE  = {mae:.3f} g/L")


# ----------------------------
# STEP 7: Predict at a new process condition
# Example: 1% inoculum and 36 h
# ----------------------------

new_point = pd.DataFrame({
    "A": [np.log10(1.0)],
    "B": [(36 - 30) / 18]
})

prediction = model.get_prediction(new_point).summary_frame()

print("\nPrediction at 1% inoculum and 36 h:")
print(prediction)


# ----------------------------
# STEP 8: Create grid for contour and 3D surface plots
# ----------------------------

a_range = np.linspace(-1, 1, 100)
b_range = np.linspace((24 - 30) / 18, (48 - 30) / 18, 100)

A_grid, B_grid = np.meshgrid(a_range, b_range)

grid = pd.DataFrame({
    "A": A_grid.ravel(),
    "B": B_grid.ravel()
})

grid["Predicted_Lactic_Acid"] = model.predict(grid)

Z = grid["Predicted_Lactic_Acid"].values.reshape(A_grid.shape)

# Convert coded values back to actual units
Inoculum_actual = 10 ** A_grid
Time_actual = (B_grid * 18) + 30


# ----------------------------
# STEP 9: Observed vs predicted plot
# ----------------------------

plt.figure(figsize=(6, 6))
plt.scatter(df["Lactic_Acid_gL"], df["Predicted_Lactic_Acid"])

min_val = min(df["Lactic_Acid_gL"].min(), df["Predicted_Lactic_Acid"].min())
max_val = max(df["Lactic_Acid_gL"].max(), df["Predicted_Lactic_Acid"].max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Observed Lactic Acid (g/L)")
plt.ylabel("Predicted Lactic Acid (g/L)")
plt.title("Observed vs Predicted Lactic Acid")
plt.tight_layout()
plt.savefig("figures/observed_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------
# STEP 10: Contour plot
# ----------------------------

plt.figure(figsize=(8, 6))

contour_filled = plt.contourf(
    Time_actual,
    Inoculum_actual,
    Z,
    levels=20
)

plt.colorbar(contour_filled, label="Predicted Lactic Acid (g/L)")

plt.contour(
    Time_actual,
    Inoculum_actual,
    Z,
    levels=10
)

plt.xlabel("Time (h)")
plt.ylabel("Inoculum (%)")
plt.yscale("log")
plt.title("Contour Plot: Lactic Acid vs Time and Inoculum")
plt.tight_layout()
plt.savefig("figures/contour_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------
# STEP 11: 3D surface plot
# ----------------------------

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    Time_actual,
    Inoculum_actual,
    Z,
    alpha=0.9
)

ax.set_xlabel("Time (h)")
ax.set_ylabel("Inoculum (%)")
ax.set_zlabel("Predicted Lactic Acid (g/L)")
ax.set_title("RSM Surface Plot: Lactic Acid vs Time and Inoculum")

plt.tight_layout()
plt.savefig("figures/surface_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------
# STEP 12: Find best predicted condition
# ----------------------------

best_idx = grid["Predicted_Lactic_Acid"].idxmax()
best_row = grid.loc[best_idx]

best_A = best_row["A"]
best_B = best_row["B"]

best_inoculum = 10 ** best_A
best_time = (best_B * 18) + 30
best_lactic = best_row["Predicted_Lactic_Acid"]

print("\nBEST PREDICTED CONDITION")
print(f"Predicted best inoculum (%)       = {best_inoculum:.3f}")
print(f"Predicted best time (h)           = {best_time:.2f}")
print(f"Predicted maximum lactic acid     = {best_lactic:.3f} g/L")
