import cv2
import numpy as np

def extract_silhouette(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return mask

def compute_vertical_ratio(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (ys.max() - ys.min()) / (xs.max() - xs.min())

def compute_volume_balance(mask):
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return ys.mean() / mask.shape[0]

def compute_straightness(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, epsilon=5, closed=True)
    return len(approx) / len(cnt)

def extract_shape_features_from_image(img):
    mask = extract_silhouette(img)

    straightness = compute_straightness(mask)
    if straightness is None:
        return None

    return {
        "vertical": compute_vertical_ratio(mask),
        "straightness": straightness,
        "curviness": 1 - straightness,
        "volume_y": compute_volume_balance(mask)
    }
