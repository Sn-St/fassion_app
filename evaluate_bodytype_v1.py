import os
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================
# ① 必要な関数をインポート
# ============================
from silhouette_inference_local import infer_silhouette
from extract_shape_features_from_image import extract_shape_features_from_image
from score_body_type import score_body_type


# ============================
# ② 1枚の画像を骨格診断する関数
# ============================
def predict_body_type(image_path):
    # --- 画像読み込み ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像が読み込めませんでした: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Roboflow シルエット推論 ---
    result = infer_silhouette(image_path)
    if result is None:
        print(f"シルエット推論に失敗: {image_path}")
        return None

    # infer_silhouette() は dict を返す
    silhouette_label = result["top1"]["label"]
    silhouette_conf = result["top1"]["confidence"]

    # --- 特徴量抽出 ---
    features = extract_shape_features_from_image(img_rgb)
    if features is None:
        print(f"特徴量が抽出できませんでした: {image_path}")
        return None

    vertical = features["vertical"]
    straightness = features["straightness"]
    curviness = features["curviness"]
    volume_y = features["volume_y"]

    # --- 最終骨格タイプ判定 ---
    scores, confidence, reason = score_body_type(
        vertical,
        straightness,
        curviness,
        volume_y,
        silhouette_label,
        silhouette_conf
    )

    # ★ 最終ラベルは scores の最大値から決める
    final_label = max(scores, key=scores.get).capitalize()

    return final_label


# ============================
# ③ 全画像を一括で評価
# ============================
def evaluate_dataset(base_dir="data_labeled_shape"):
    results = []

    for true_label in ["straight", "wave", "natural"]:
        folder = os.path.join(base_dir, true_label)

        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, filename)

                pred = predict_body_type(path)

                results.append({
                    "image": filename,
                    "true": true_label.capitalize(),
                    "pred": pred.capitalize() if pred else "Error"
                })

    df = pd.DataFrame(results)
    df.to_csv("bodytype_eval_results.csv", index=False)
    print("保存しました：bodytype_eval_results.csv")

    return df


# ============================
# ④ 精度計算
# ============================
def evaluate_accuracy(df):
    y_true = df["true"]
    y_pred = df["pred"]

    print("\n===== Accuracy =====")
    print(accuracy_score(y_true, y_pred))

    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred))

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(y_true, y_pred))


# ============================
# ⑤ 実行
# ============================
if __name__ == "__main__":
    df = evaluate_dataset("data_labeled_shape")
    evaluate_accuracy(df)