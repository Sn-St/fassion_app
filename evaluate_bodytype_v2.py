import os
import cv2
import pandas as pd
import numpy as np

from silhouette_inference_local import infer_silhouette
from extract_shape_features_from_image import extract_shape_features_from_image
from score_body_type import score_body_type


def predict_scores(image_path):
    """画像1枚に対してAIの骨格スコアを返す"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # シルエット推論
    silhouette = infer_silhouette(image_path)
    silhouette_label = silhouette["top1"]["label"]
    silhouette_conf = silhouette["top1"]["confidence"]

    # 特徴量抽出
    features = extract_shape_features_from_image(img_rgb)
    vertical = features["vertical"]
    straightness = features["straightness"]
    curviness = features["curviness"]
    volume_y = features["volume_y"]

    # スコア計算
    scores, confidence, reason = score_body_type(
        vertical,
        straightness,
        curviness,
        volume_y,
        silhouette_label,
        silhouette_conf
    )

    return scores


def evaluate(evaluation_csv, image_folder):
    """CSVの正解スコアとAI予測スコアを比較して評価する"""
    df = pd.read_csv(evaluation_csv)

    results = []

    for _, row in df.iterrows():
        image_name = row["image"]
        true_scores = np.array([row["straight"], row["wave"], row["natural"]])

        image_path = os.path.join(image_folder, image_name)
        pred = predict_scores(image_path)
        pred_scores = np.array([
            pred["straight"],
            pred["wave"],
            pred["natural"]
        ])

        # 誤差計算
        mae = np.mean(np.abs(pred_scores - true_scores))
        rmse = np.sqrt(np.mean((pred_scores - true_scores) ** 2))

        results.append([
            image_name,
            row["straight"], row["wave"], row["natural"],   # true

            pred_scores[0], pred_scores[1], pred_scores[2], # pred

            mae, rmse
        ])


    # 結果をまとめる
    result_df = pd.DataFrame(
        results,
        columns=[
            "image",
            "straight_true", "wave_true", "natural_true",
            "straight_pred", "wave_pred", "natural_pred",
            "mae", "rmse"
        ]
    )

    print(result_df)
    print("\n=== Summary ===")
    print(result_df.describe())

    # 保存
    result_df.to_csv("evaluation_results_v2.csv", index=False)
    print("\n保存しました: evaluation_results_v2.csv")


if __name__ == "__main__":
    # 実行例（必要に応じて変更）
    evaluate("evaluation_labels.csv", "evaluation_images")