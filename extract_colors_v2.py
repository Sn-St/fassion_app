import numpy as np
from skimage import color
from PIL import Image

def extract_average_color_from_file(uploaded_file):
    """
    Streamlit uploaded_file から平均色を抽出
    """
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image) / 255.0

    # Lab
    lab = color.rgb2lab(img_np)
    L = lab[:, :, 0].mean()
    a = lab[:, :, 1].mean()
    b = lab[:, :, 2].mean()

    # HSV
    hsv = color.rgb2hsv(img_np)
    H = hsv[:, :, 0].mean() * 360
    S = hsv[:, :, 1].mean()
    V = hsv[:, :, 2].mean()

    return {
        "lab": {"L": float(L), "a": float(a), "b": float(b)},
        "hsv": {"hue": float(H), "saturation": float(S), "value": float(V)}
    }
