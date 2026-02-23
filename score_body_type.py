import math

# =========================
# ユーティリティ
# =========================
def clamp(x, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, x))


# =========================
# シルエット → 骨格 対応表
# （7クラス → 3タイプ）
# =========================
SILHOUETTE_TO_BODY = {
    # 直線・細身
    "Straight": "straight",
    "Slim": "straight",

    # 下重心・広がり
    "Volume": "wave",
    "Flare": "wave",

    # 余白・箱型
    "Box": "natural",
    "Oversize": "natural",
    "Wide": "natural",
}


# =========================
# ① 特徴量 → 言語化
# =========================
def describe_vertical_ratio(v):
    if v >= 0.75:
        return "縦にすっきりした印象が強い"
    elif v >= 0.55:
        return "縦横のバランスが中庸"
    else:
        return "横方向に広がりやすいシルエット"

def describe_straightness(s):
    if s >= 0.75:
        return "直線的でシャープなラインが目立つ"
    elif s >= 0.55:
        return "直線と曲線がバランスよく混在"
    else:
        return "直線的なラインは控えめ"

def describe_curviness(c):
    if c >= 0.75:
        return "曲線的で柔らかい印象が強い"
    elif c >= 0.55:
        return "適度に丸みがある"
    else:
        return "丸みは控えめ"

def describe_volume_balance(vb):
    if vb >= 0.60:
        return "重心がやや下に寄る傾向"
    elif vb >= 0.45:
        return "重心が中央に近い"
    else:
        return "重心が上に寄りやすい"


# =========================
# ② confidence の説明
# =========================
def describe_confidence(conf):
    if conf >= 0.35:
        return "今回の診断は比較的はっきりした傾向が見られ、安定した結果となっています。"
    elif conf >= 0.18:
        return "複数のタイプの特徴が混在しており、どちらにも当てはまる部分があります。"
    else:
        return "タイプ間の差が小さく、今回の診断には揺らぎが見られます。参考程度にご覧ください。"


# =========================
# ③ 骨格タイプごとの注目順
# =========================
def prioritize_features(body_type, features):
    if body_type == "Straight":
        order = ["straightness", "vertical_ratio", "volume_balance", "curviness"]
    elif body_type == "Wave":
        order = ["curviness", "volume_balance", "vertical_ratio", "straightness"]
    else:  # Natural
        order = ["straightness", "curviness", "volume_balance", "vertical_ratio"]

    return sorted(features.items(), key=lambda x: order.index(x[0]))


# =========================
# ④ 理由文生成（AI統合）
# =========================
def generate_reason_text(body_type, features, confidence, silhouette_info=None):
    descriptions = {
        "vertical_ratio": describe_vertical_ratio(features["vertical_ratio"]),
        "straightness": describe_straightness(features["straightness"]),
        "curviness": describe_curviness(features["curviness"]),
        "volume_balance": describe_volume_balance(features["volume_balance"]),
    }

    ordered = prioritize_features(body_type, features)
    top_two = [descriptions[name] for name, _ in ordered[:2]]

    # Naturalは表現を少し安定させる
    if body_type == "Natural":
        top_two = [
            "直線と曲線がバランスよく混在している",
            "重心が中央に近い"
        ]

    confidence_text = describe_confidence(confidence)

    silhouette_text = ""
    if silhouette_info:
        silhouette_text = (
            f"\n\n【シルエット分析（AI）】\n"
            f"衣服のシルエットは「{silhouette_info['label']}」と判定され、"
            f"骨格診断では「{silhouette_info['mapped_body']}タイプ寄り」の補助情報として考慮されています。"
        )

    reason = (
        f"【特徴の説明】\n"
        f"{top_two[0]}傾向があります。また、{top_two[1]}傾向が見られます。\n\n"
        f"【診断のまとめ】\n"
        f"これらの特徴から、全体として {body_type} タイプに近い印象となっています。\n"
        f"{confidence_text}"
        f"{silhouette_text}"
    )

    return reason


# =========================
# ⑤ メイン：スコア計算 + AI加点 + 補正係数
# =========================
def score_body_type(
    vertical,
    straightness,
    curviness,
    volume_y,
    silhouette_label=None,
    silhouette_conf=0.0
):
    scores = {
        "straight": 0.0,
        "wave": 0.0,
        "natural": 0.0
    }

    # ---- 既存ルールベース ----
    scores["straight"] += straightness * 0.4
    scores["straight"] += clamp((vertical - 1.0) * 0.5) * 0.3
    scores["straight"] += clamp(0.6 - volume_y) * 0.3

    scores["wave"] += curviness * 0.5
    scores["wave"] += clamp(volume_y - 0.4) * 0.3
    scores["wave"] += clamp(1.0 - vertical) * 0.2

    balance = 1.0 - abs(volume_y - 0.5) * 2
    scores["natural"] += balance * 0.4
    scores["natural"] += clamp(1.0 - abs(vertical - 1.1)) * 0.3
    scores["natural"] += clamp(1.0 - abs(straightness - curviness)) * 0.3

    # ---- AIシルエット加点 ----
    silhouette_info = None
    if silhouette_label and silhouette_conf >= 0.7:
        mapped = SILHOUETTE_TO_BODY.get(silhouette_label)
        if mapped:
            scores[mapped] += 0.25
            silhouette_info = {
                "label": silhouette_label,
                "confidence": silhouette_conf,
                "mapped_body": mapped
            }

    # ---- 正規化 ----
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total

    # =========================
    # ★ 補正係数（キャリブレーション）
    # =========================
    scores["straight"] += 0.48
    scores["wave"] += 0.38
    scores["natural"] += 0.28

    # 0〜1 にクリップ
    for k in scores:
        scores[k] = clamp(scores[k])

    # confidence（1位と2位の差）
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = clamp(sorted_scores[0] - sorted_scores[1])

    body_type = max(scores, key=scores.get).capitalize()

    features = {
        "vertical_ratio": vertical,
        "straightness": straightness,
        "curviness": curviness,
        "volume_balance": volume_y
    }

    reason = generate_reason_text(
        body_type,
        features,
        confidence,
        silhouette_info
    )

    return scores, confidence, reason