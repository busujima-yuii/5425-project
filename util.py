# 以punctuation为终止符将片段合并, 不足2秒的片段会自动合并到下一个片段
def merge_segments(segments, min_seg_duration=2.0):
    merged = []
    buf = segments[0].copy()

    for seg in segments[1:]:
        seg_dur = seg["end"] - seg["start"]
        ends_with_punc = buf["text"].strip().endswith(('.', '?', '!'))
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
    if text.startswith('{') and text.endswith('}'):
        return text

    # 去除开头的 ```json 或 ```
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    # 去除结尾的 ```
    if text.endswith("```"):
        text = text[:-3].strip()

    raise ValueError("No valid JSON found in GPT response.")

def clean_and_merge_segments(segments: List[Dict], min_chars=10, time_gap=1.0) -> List[Dict]:
    """
    1. 合并相邻重复内容（只要文本一样就合并时间）
    2. 清洗空文本或无意义内容（如 '...', '.', 'uh', '')
    """
    cleaned = []
    for seg in segments:
        text = seg['text'].strip()
        if len(text) < min_chars or text in {"...", ".", "", "uh", "um"}:
            continue

        if cleaned and cleaned[-1]['text'].strip().lower() == text.lower():
            # 合并时间段
            cleaned[-1]['end'] = seg['end']
        else:
            cleaned.append(seg)

    return cleaned

def compress_image(image_path, max_size=(512, 512)):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, max_size)
    temp_path = image_path.replace(".jpg", "_compressed.jpg")
    cv2.imwrite(temp_path, resized)
    return temp_path