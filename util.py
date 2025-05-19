# 以punctuation为终止符将片段合并, 不足2秒的片段会自动合并到下一个片段
def merge_segments(segments, min_seg_duration=2.0):
    merged = []
    buf = segments[0].copy()

    for seg in segments[1:]:
        seg_dur = seg["end"] - seg["start"]
        ends_with_punc = buf["text"].strip().endswith((".", "?", "!"))
        if ends_with_punc and seg_dur >= min_seg_duration:
            merged.append(buf.copy())
            buf = seg.copy()
        else:
            buf["end"] = seg["end"]
            buf["text"] += seg["text"]
    merged.append(buf.copy())
    return merged


import re
from typing import Dict, List

import cv2
import numpy as np


def extract_json_from_response(text: str) -> str:
    """
    从 GPT 回复中提取 JSON 内容，支持 ```json 和 ``` 包裹的 markdown 块
    """
    # 优先匹配 ```json 块
    match = re.search(r"```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```", text)
    if match:
        text = match.group(1).strip()

    # 如果没有 markdown 包裹，直接尝试整段作为 json
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    # 去除开头的 ```json 或 ```
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()

    # 去除结尾的 ```
    if text.endswith("```"):
        text = text[:-3].strip()

    raise ValueError("No valid JSON found in GPT response.")


def clean_and_merge_segments(
    segments: List[Dict], min_chars=10, time_gap=1.0
) -> List[Dict]:
    """
    1. 合并相邻重复内容（只要文本一样就合并时间）
    2. 清洗空文本或无意义内容（如 '...', '.', 'uh', '')
    """
    cleaned = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) < min_chars or text in {"...", ".", "", "uh", "um"}:
            continue

        if cleaned and cleaned[-1]["text"].strip().lower() == text.lower():
            # 合并时间段
            cleaned[-1]["end"] = seg["end"]
        else:
            cleaned.append(seg)

    return cleaned


def compress_image(image_path, max_size=(512, 512)):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, max_size)
    temp_path = image_path.replace(".jpg", "_compressed.jpg")
    cv2.imwrite(temp_path, resized)
    return temp_path

def hue_to_color_name(hue: int) -> str:
    color_ranges = {
        (0, 10): "Red",
        (11, 20): "Orange-Red",
        (21, 30): "Orange",
        (31, 40): "Yellow-Orange",
        (41, 60): "Yellow",
        (61, 80): "Yellow-Green",
        (81, 100): "Green",
        (101, 120): "Cyan-Green",
        (121, 140): "Cyan",
        (141, 160): "Blue-Cyan",
        (161, 180): "Blue",
    }
    for (lower, upper), color in color_ranges.items():
        if lower <= hue <= upper:
            return color
    return "Unknown"

def analyse_image_colors(frame: np.ndarray) -> Dict:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180]).flatten()
    hist /= hist.sum()
    dominant_indices = np.argsort(hist)[-5:][::-1]
    dominant_colors = []
    for idx in dominant_indices:
        hue = idx
        percentage = float(hist[idx] * 100)
        color_name = hue_to_color_name(hue)
        dominant_colors.append(
            {
                "color": color_name,
                "hue": int(hue),
                "percentage": round(percentage, 2),
            }
        )
    return {
        "dominant_colors": dominant_colors,
        "color_diversity": float(np.sum(hist > 0.01) / len(hist)),
    }

def compute_ssim(img1, img2):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        C1 = 6.5025
        C2 = 58.5225
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()