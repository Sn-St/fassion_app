import os
import joblib
import numpy as np

# =========================
# モデルパス（このファイル基準）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "season_model_rf_proba.pkl")

FEATURES = [
    "L", "a", "b",
    "hue", "saturation", "value",
    "chroma", "cool_warm"
]

# モデル読み込み
model = joblib.load(MODEL_PATH)

# =========================
# 推論関数
# =========================
def infer_season(color_features):
    x = [
        color_features["lab"]["L"],
        color_features["lab"]["a"],
        color_features["lab"]["b"],
        color_features["hsv"]["hue"],
        color_features["hsv"]["saturation"],
        color_features["hsv"]["value"],
        np.sqrt(color_features["lab"]["a"]**2 + color_features["lab"]["b"]**2),  # chroma
        color_features["lab"]["b"] - color_features["lab"]["a"]                 # cool_warm
    ]

    proba = model.predict_proba([x])[0]
    classes = model.classes_

    # 上位2クラス
    idx = np.argsort(proba)[::-1]
    top1 = (classes[idx[0]], float(proba[idx[0]]))
    top2 = (classes[idx[1]], float(proba[idx[1]]))

    return top1, top2

print("MODEL_PATH:", MODEL_PATH)
