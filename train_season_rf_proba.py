import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# =========================
# データ読み込み
# =========================
df = pd.read_csv("color_train.csv")
df["season"] = df["season"].str.strip().str.capitalize()

# =========================
# 特徴量
# =========================
df["chroma"] = np.sqrt(df["a"]**2 + df["b"]**2)
df["cool_warm"] = df["b"] - df["a"]

FEATURES = [
    "L", "a", "b",
    "hue", "saturation", "value",
    "chroma", "cool_warm"
]

df = df.dropna(subset=FEATURES + ["season"])

X = df[FEATURES]
y = df["season"]

# =========================
# Random Forest（確定版）
# =========================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=9,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, "season_model_rf_proba.pkl")
print("✅ RF model with proba saved")
