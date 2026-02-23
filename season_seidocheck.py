import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# =========================
# ① データ読み込み
# =========================
df = pd.read_csv("color_train.csv")
df["season"] = df["season"].str.strip().str.capitalize()

# =========================
# ② 特徴量作成
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
# ③ train/test split（重要）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# ④ RandomForest モデル学習
# =========================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=9,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# ⑤ 精度評価（Accuracy / Report / Confusion Matrix）
# =========================
y_pred = model.predict(X_test)

print("===== Accuracy =====")
print(accuracy_score(y_test, y_pred))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

# =========================
# ⑥ HSV・特徴量の分布可視化
# =========================
df[["hue", "saturation", "value"]].hist(bins=30, figsize=(10, 4))
plt.suptitle("HSV Distribution")
plt.tight_layout()
plt.show()

# =========================
# ⑦ 季節ごとの特徴量の違い（pairplot）
# =========================
sns.pairplot(
    df,
    vars=["hue", "saturation", "value", "chroma"],
    hue="season",
    diag_kind="hist"
)
plt.show()

# =========================
# ⑧ 特徴量重要度（Feature Importance）
# =========================
importance = pd.Series(model.feature_importances_, index=FEATURES)
print("\n===== Feature Importance =====")
print(importance.sort_values(ascending=False))

importance.sort_values().plot(kind="barh", figsize=(6, 4))
plt.title("Feature Importance")
plt.show()

# =========================
# ⑨ モデル保存（任意）
# =========================
joblib.dump(model, "season_model_rf_proba.pkl")
print("\nModel saved: season_model_rf_proba.pkl")