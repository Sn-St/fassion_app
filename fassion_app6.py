import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

# =========================
# ページ設定
# =========================
st.set_page_config(page_title="Style Diagnosis", layout="wide")
st.title("👗 Color & Shape Finder")

# =========================
# CSS（太め下線・上品ボックス・色ドット・色バー）
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 18px;
}

/* 見出し */
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.4rem !important; }

/* 太めで文字に少し被る下線（2行対応OK） */
.underline-gray {
    display: inline;
    background: linear-gradient(to bottom, transparent 55%, #e5e5e5 55%);
    font-weight: 700;
    font-size: 1.6rem;
    line-height: 1.6;
}

/* 上品な説明ボックス */
.explain-box {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    padding: 18px 22px;
    border-radius: 10px;
    margin: 20px 0;
}
.explain-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 8px;
}

/* カラードット（パーソナルカラー用） */
.color-dot {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 6px;
    border: 1px solid #ccc;
}

/* グラデーションバー */
.grad-bar {
    height: 18px;
    border-radius: 8px;
    margin: 6px 0 14px 0;
    position: relative;
}
.grad-a {
    background: linear-gradient(to right, green, white, red);
}
.grad-b {
    background: linear-gradient(to right, blue, white, yellow);
}
.grad-marker {
    width: 10px;
    height: 18px;
    background: #222;
    position: absolute;
    top: 0;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# サイドバー
# =========================
st.sidebar.title("🔍 表示切り替え")

view_mode = st.sidebar.radio(
    "表示モード",
    ["診断結果", "詳細分析"]
)

detail_target = None
if view_mode == "詳細分析":
    detail_target = st.sidebar.radio(
        "詳細項目",
        ["パーソナルカラー", "骨格タイプ"]
    )

# =========================
# 画像アップロード
# =========================
uploaded_file = st.file_uploader(
    "🖼 診断したい画像をアップロード",
    type=["png", "jpg", "jpeg"]
)

if not uploaded_file:
    st.info("画像をアップロードしてください")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_np = np.array(image)

# =========================
# 外部処理
# =========================
#from extract_colors_v2 import extract_average_color_from_file
from extract_shape_features_from_image import extract_shape_features_from_image
from silhouette_inference_local import infer_silhouette
from score_body_type import score_body_type
from season_inference import infer_season

#color_features = extract_average_color_from_file(uploaded_file)
shape_features = extract_shape_features_from_image(img_np)
sil = infer_silhouette(uploaded_file)

# ============================
# ★ 服の色抽出に必要な関数（必ずこの位置に置く）
# ============================
import mediapipe as mp
import cv2
import numpy as np
from skimage import color

def extract_clothes_color_mediapipe(image):
    img_np = np.array(image)
    h, w, _ = img_np.shape

    img_rgb = img_np

    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = mp_selfie.process(img_rgb)

    mask = results.segmentation_mask
    mask = cv2.resize(mask, (w, h))

    inv_mask = 1.0 - mask
    clothes_region = img_np * inv_mask[:, :, None]

    nonzero = clothes_region[np.any(clothes_region != 0, axis=2)]

    if len(nonzero) == 0:
        y1, y2 = int(h*0.35), int(h*0.85)
        x1, x2 = int(w*0.25), int(w*0.75)
        crop = img_np[y1:y2, x1:x2]
        avg = crop.mean(axis=(0,1))
    else:
        avg = nonzero.mean(axis=0)

    return avg


# --- 色変換関数（ここに置くのが正しい） ---
def rgb_to_hsv(rgb):
    r, g, b = rgb / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if mx == 0 else diff / mx
    v = mx

    return {"hue": h, "saturation": s, "value": v}

def rgb_to_lab(rgb):
    rgb_norm = np.array(rgb) / 255.0
    lab = color.rgb2lab([[rgb_norm]])[0][0]
    return {"L": lab[0], "a": lab[1], "b": lab[2]}


# =========================
# ★ 服の色を抽出（Mediapipe）
# =========================
clothes_rgb = extract_clothes_color_mediapipe(image)
clothes_hsv = rgb_to_hsv(clothes_rgb)
clothes_lab = rgb_to_lab(clothes_rgb)

# ML に渡す色を「服の色」に変更
color_features = {"hsv": clothes_hsv, "lab": clothes_lab}


# =========================
# パーソナルカラー（ML）
# =========================
(top_season, top_conf), (sub_season, sub_conf) = infer_season(color_features)

# マッピング用にも服の色を使う
hsv = clothes_hsv
lab = clothes_lab

# 日本語表記
season_jp = {
    "Spring": "スプリング（春）",
    "Summer": "サマー（夏）",
    "Autumn": "オータム（秋）",
    "Winter": "ウィンター（冬）"
}

body_jp = {
    "straight": "ストレート",
    "wave": "ウェーブ",
    "natural": "ナチュラル"
}

# =========================
# メイン：診断結果（太め下線）
# =========================
if view_mode == "診断結果":
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("🖼 入力画像")
        st.image(image, width=260)

    with col2:
        st.subheader("✨ 診断結果")

        st.markdown(
            f'<span class="underline-gray">🎨 パーソナルカラー：{season_jp[top_season]}</span>',
            unsafe_allow_html=True
        )
        st.caption(f"信頼度：{top_conf:.2f}")

        if top_conf < 0.55:
            st.write(f"※{season_jp[sub_season]} の可能性もあります（{sub_conf:.2f}）")

        scores, confidence, _ = score_body_type(
            shape_features["vertical"],
            shape_features["straightness"],
            shape_features["curviness"],
            shape_features["volume_y"],
            silhouette_label=sil["top1"]["label"],
            silhouette_conf=sil["top1"]["confidence"]
        )

        body_type = max(scores, key=scores.get)

        st.markdown(
            f'<span class="underline-gray">🧍‍♀️ 骨格タイプ：{body_jp[body_type]}</span>',
            unsafe_allow_html=True
        )
        st.caption(f"信頼度：{confidence:.2f}")

        st.markdown("""
        <div style="font-size:16px; color:#555; margin-top:20px;">
        ※信頼度が低い場合でも、診断が間違っているという意味ではありません。<br>
        特徴が複数のタイプにまたがっていたり、どのタイプにも当てはまりやすい場合に、
        信頼度が低く表示されることがあります。
        </div>
        """, unsafe_allow_html=True)

# =========================
# 詳細分析（上品ボックス＋色ドット版）
# =========================
else:

    # =========================
    # パーソナルカラー詳細
    # =========================
    if detail_target == "パーソナルカラー":
        st.subheader("🎨 パーソナルカラー詳細")

        import streamlit.components.v1 as components

        html_block = """
        <div style="
            background:#fafafa;
            border:1px solid #ddd;
            border-radius:12px;
            padding:24px;
            font-size:1.25rem;
            line-height:1.9;
        ">
            <div style="font-size:1.6rem; font-weight:700; margin-bottom:14px;">
                🎨 パーソナルカラーとは？
            </div>

            肌・髪・瞳の色の特徴から、似合いやすい色のグループを分類したものです。<br>
            服・メイク・アクセサリーの色選びの指標になります。

            <div style="margin-top:22px;">

                <!-- スプリング -->
                <div style="margin-bottom:14px;">
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#F7D9A6;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#FFBFA8;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#F7E86A;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:10px;border:1px solid #ccc;background:#C8E6A0;"></span>
                    スプリング：明るくクリア、黄み寄り
                </div>

                <!-- サマー -->
                <div style="margin-bottom:14px;">
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#DCD9F7;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#F3CCE0;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#CDE7F5;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:10px;border:1px solid #ccc;background:#BFC7D9;"></span>
                    サマー：柔らかく涼しげ、青み寄り
                </div>

                <!-- オータム -->
                <div style="margin-bottom:14px;">
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#D6B88A;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#C97A4A;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#A8B86E;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:10px;border:1px solid #ccc;background:#7A8B52;"></span>
                    オータム：深みのある暖色、落ち着いた色
                </div>

                <!-- ウィンター -->
                <div style="margin-bottom:14px;">
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#BFC7FF;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#D94CFF;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:4px;border:1px solid #ccc;background:#000000;"></span>
                    <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                                 margin-right:10px;border:1px solid #ccc;background:#E6E6E6;"></span>
                    ウィンター：コントラスト強め、鮮やかでクール
                </div>

            </div>
        </div>
        """

        components.html(html_block, height=500)


        # ============================
        # レイアウト開始
        # ============================
        col_a, col_b = st.columns([1, 1.4])

        with col_a:
            st.subheader("🖼 入力画像")
            st.image(image, width=260)

        with col_b:
            st.markdown("### ML診断結果")
            st.write(
                f"""
        - **パーソナルカラー**：**{season_jp[top_season]}**
        - **信頼度**：{top_conf:.2f}
        """
            )

            st.caption("""
        ※AI診断は服の色を8つの指標から総合的に判定しています。
        　明度×彩度マッピングは2つの指標のみを使うため、結果が完全に一致しないことがあります。
        """)


        # ============================
        # 明度 × 彩度 マッピング
        # ============================
        st.markdown("### 明度 × 彩度 マッピング")


        df_hsv = pd.DataFrame({
            "彩度": [hsv["saturation"]],
            "明度": [hsv["value"]],
        })

        background = pd.DataFrame([
            {"x1": 0.5, "x2": 1.0, "y1": 0.5, "y2": 1.0, "color": "#FFD27F"},
            {"x1": 0.0, "x2": 0.5, "y1": 0.5, "y2": 1.0, "color": "#C9C6FF"},
            {"x1": 0.5, "x2": 1.0, "y1": 0.0, "y2": 0.5, "color": "#D8A86A"},
            {"x1": 0.0, "x2": 0.5, "y1": 0.0, "y2": 0.5, "color": "#AFC8FF"},
        ])

        rect = (
            alt.Chart(background)
            .mark_rect(opacity=0.55)
            .encode(
                x=alt.X("x1:Q", bin="binned", scale=alt.Scale(domain=[0, 1])),
                x2="x2:Q",
                y=alt.Y("y1:Q", bin="binned", scale=alt.Scale(domain=[0, 1])),
                y2="y2:Q",
                color=alt.Color("color:N", scale=None)
            )
        )

        labels = pd.DataFrame([
            {"x": 0.75, "y": 0.85, "text": "Spring"},
            {"x": 0.25, "y": 0.85, "text": "Summer"},
            {"x": 0.75, "y": 0.15, "text": "Autumn"},
            {"x": 0.25, "y": 0.15, "text": "Winter"},
        ])

        label_layer = (
            alt.Chart(labels)
            .mark_text(font="Hiragino Sans", fontSize=22, color="#444", opacity=0.35)
            .encode(x="x:Q", y="y:Q", text="text:N")
        )

        cross = (
            alt.Chart(pd.DataFrame({"x": [0.5]}))
            .mark_rule(color="#666", strokeWidth=1.2, opacity=0.7)
            .encode(x="x:Q")
            +
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(color="#666", strokeWidth=1.2, opacity=0.7)
            .encode(y="y:Q")
        )

        point = (
            alt.Chart(df_hsv)
            .mark_text(
                text="★",
                font="Hiragino Sans",
                fontSize=40,
                color="gold",
                stroke="white",
                strokeWidth=2,
                opacity=0.98
            )
            .encode(
                x=alt.X("彩度:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("明度:Q", scale=alt.Scale(domain=[0, 1])),
            )
        )

        chart = (
            (rect + label_layer + cross + point)
            .properties(width=320, height=320, title="服の色の位置（明度 × 彩度）")
        )

        st.altair_chart(chart, use_container_width=False)

        with st.expander("ℹ️ AI診断とマッピングの違いについて"):
            st.markdown("""
        AI診断は服の色を **8つの指標**（L・a・b・色相・彩度・明度・色の強さ・暖色/寒色バランス）に分解し、
        総合的に季節タイプを判定しています。

        一方、明度×彩度マッピングは **2つの指標（明度と彩度）だけ** を使った
        “色の位置を直感的に見るための地図” です。

        そのため、**両者が完全に一致しないことがありますが、どちらも服の色を基にしています。**
        """)
            
        # ============================
        # 色の特徴量
        # ============================
        st.markdown("### 色の特徴量（数値とバー表示）")

        col_l, col_hsv = st.columns(2)

        with col_l:
            st.markdown("**L（明度：見た目の明るさ）**")
            st.write(f"{lab['L']:.1f} / 100")
            st.progress(float(min(max(lab["L"] / 100, 0), 1)))

            st.markdown("**Hue（色相：色の方向）**")
            st.write(f"{hsv['hue']:.1f}°")
            st.progress(float(min(max(hsv["hue"] / 360, 0), 1)))

        with col_hsv:
            st.markdown("**Saturation（彩度：色の鮮やかさ）**")
            st.write(f"{hsv['saturation']:.2f}")
            st.progress(float(min(max(hsv["saturation"], 0), 1)))

            st.markdown("**Value（明るさ：光の強さとしての明るさ）**")
            st.write(f"{hsv['value']:.2f}")
            st.progress(float(min(max(hsv["value"], 0), 1)))

        st.markdown("### 色のバランス（赤み・緑み／黄み・青み）")

        a_pos = (lab["a"] + 20) / 40 * 100
        b_pos = (lab["b"] + 20) / 40 * 100

        st.markdown(f"**a（緑み ↔ 赤み）**：{lab['a']:.2f}")
        st.markdown(f"""
        <div class="grad-bar grad-a">
            <div class="grad-marker" style="left:{a_pos}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**b（青み ↔ 黄み）**：{lab['b']:.2f}")
        st.markdown(f"""
        <div class="grad-bar grad-b">
            <div class="grad-marker" style="left:{b_pos}%"></div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # 骨格タイプ詳細
    # =========================
    if detail_target == "骨格タイプ":
        st.subheader("🧍 骨格タイプ詳細")

        st.markdown("""
        <div class="explain-box">
            <div class="explain-title">🧍 骨格タイプとは？</div>
            体のライン・質感・重心バランスから、似合いやすい服のシルエットを分類したものです。

        - **ストレート**：立体的でメリハリ。シンプルで上質な服が似合う。
        - **ウェーブ**：華奢で柔らかい。軽やかでフィット感のある服が似合う。
        - **ナチュラル**：骨感がありスタイリッシュ。ラフでゆったりした服が似合う。

        </div>
        """, unsafe_allow_html=True)


        scores, confidence, reason = score_body_type(
            shape_features["vertical"],
            shape_features["straightness"],
            shape_features["curviness"],
            shape_features["volume_y"],
            silhouette_label=sil["top1"]["label"],
            silhouette_conf=sil["top1"]["confidence"]
        )

        labels_jp = ["ストレート", "ウェーブ", "ナチュラル"]
        values = [scores["straight"], scores["wave"], scores["natural"]]

        df = pd.DataFrame({
            "タイプ": labels_jp,
            "スコア": values
        })

        best_idx = values.index(max(values))
        df["表示名"] = [
            f"{labels_jp[i]} ★" if i == best_idx else labels_jp[i]
            for i in range(3)
        ]

        st.markdown("### 骨格タイプのバランス（スコア）")

        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("スコア:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("表示名:N", sort="-x"),
                color=alt.Color("スコア:Q", scale=alt.Scale(scheme="tealblues"))
            )
            .properties(height=220)
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(f"信頼度：{confidence:.2f}")

        st.markdown("### 🔍 骨格タイプの分析ポイント")
        st.info(reason)