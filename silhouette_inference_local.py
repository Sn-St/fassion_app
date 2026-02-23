# silhouette_inference_local.py
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os   # ← ★これを追加

# ===== 設定 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ← ★追加
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")         # ← ★ここを変更
IMG_SIZE = 224

# モデル読み込み（1回だけ）
model = YOLO(MODEL_PATH)

def infer_silhouette(image_file):
    """
    服のシルエットを top2 + gap で返す
    """
    img = Image.open(image_file).convert("RGB")

    results = model.predict(
        source=img,
        imgsz=IMG_SIZE,
        verbose=False
    )

    probs = results[0].probs
    if probs is None:
        return None

    scores = probs.data.cpu().numpy()
    labels = results[0].names

    # 上位2クラス取得
    top_idx = np.argsort(scores)[::-1][:2]

    idx1, idx2 = top_idx[0], top_idx[1]

    top1_conf = float(scores[idx1])
    top2_conf = float(scores[idx2])

    return {
        "top1": {
            "label": labels[idx1],
            "confidence": top1_conf
        },
        "top2": {
            "label": labels[idx2],
            "confidence": top2_conf
        },
        "gap": top1_conf - top2_conf
    }
