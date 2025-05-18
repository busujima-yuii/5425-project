import os
import cv2
import json
import base64
import numpy as np
import openai
from typing import List, Dict, Tuple
from datetime import datetime
from PIL import Image

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

class VisualAnalyser:
    def __init__(self, output_folder="keyframes"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def compute_ssim(self, img1, img2):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        C1 = 6.5025
        C2 = 58.5225
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def extract_keyframes(self, video_path: str, ssim_threshold=0.5, min_frame_gap=240, max_keyframes=None, scoring='ssim') -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        success, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 1
        last_frame_idx = -1
        candidates = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_score = self.compute_ssim(prev_gray, gray_frame)

            if ssim_score < ssim_threshold and (frame_idx - last_frame_idx) >= min_frame_gap:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                score = 1.0 - ssim_score if scoring == 'ssim' else 0  # more different means more interesting
                candidates.append({
                    "timestamp": round(timestamp, 2),
                    "frame": frame,
                    "score": score,
                    "frame_idx": frame_idx
                })
                last_frame_idx = frame_idx

            prev_gray = gray_frame
            prev_frame = frame
            frame_idx += 1

        cap.release()

        if max_keyframes and len(candidates) > max_keyframes:
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:max_keyframes]
            candidates = sorted(candidates, key=lambda x: x['frame_idx'])  # restore chronological order

        keyframes = []
        for i, cand in enumerate(candidates):
            fname = f"keyframe_{i}.jpg"
            fpath = os.path.join(self.output_folder, fname)
            cv2.imwrite(fpath, cand["frame"])
            keyframes.append({
                "timestamp": cand["timestamp"],
                "filepath": fpath,
                "frame": cand["frame"]
            })

        return keyframes
    def analyse_frame_colors(self, frame: np.ndarray) -> Dict:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180]).flatten()
        hist /= hist.sum()
        dominant_indices = np.argsort(hist)[-5:][::-1]
        dominant_colors = []
        for idx in dominant_indices:
            hue = idx
            percentage = float(hist[idx] * 100)
            color_name = self.hue_to_color_name(hue)
            dominant_colors.append({"color": color_name, "hue": int(hue), "percentage": round(percentage, 2)})
        return {
            "dominant_colors": dominant_colors,
            "color_diversity": float(np.sum(hist > 0.01) / len(hist))
        }

    def hue_to_color_name(self, hue: int) -> str:
        color_ranges = {
            (0, 10): "Red", (11, 20): "Orange-Red", (21, 30): "Orange",
            (31, 40): "Yellow-Orange", (41, 60): "Yellow", (61, 80): "Yellow-Green",
            (81, 100): "Green", (101, 120): "Cyan-Green", (121, 140): "Cyan",
            (141, 160): "Blue-Cyan", (161, 180): "Blue"
        }
        for (lower, upper), color in color_ranges.items():
            if lower <= hue <= upper:
                return color
        return "Unknown"

    def describe_image_with_gpt(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You describe visual content in clear and short English."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the main objects and scene in this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()

    def get_theme_analysis(self, color_distribution: Dict[str, float]) -> Dict:
        prompt = f"""Based on the following color distribution in a video, analyze the potential theme, mood, and genre.
        Provide your analysis in JSON format with the following fields:
        - theme: main theme or subject matter
        - mood: emotional atmosphere
        - genre: potential video genre
        - score: 0-1 indicating confidence in the analysis
        - reasoning: brief explanation of your analysis

        Color Distribution:
        {json.dumps(color_distribution, indent=2)}

        Respond in JSON format only."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a video analysis expert specialising in color theory and visual storytelling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return {
                "theme": "unknown",
                "mood": "unknown",
                "genre": "unknown",
                "confidence_score": 0,
                "reasoning": "Error in theme analysis"
            }

    def analyse_video(self, video_path: str) -> List[Dict]:
        keyframes = self.extract_keyframes(video_path)
        analysed_keyframes = []
        color_distribution = {}

        for kf in keyframes:
            color_analysis = self.analyse_frame_colors(kf["frame"])
            description = self.describe_image_with_gpt(kf["filepath"])

            for color_info in color_analysis["dominant_colors"]:
                color = color_info["color"]
                if color not in color_distribution:
                    color_distribution[color] = 0
                color_distribution[color] += 1

            analysed_keyframes.append({
                "timestamp": kf["timestamp"],
                "image_path": kf["filepath"],
                "color_analysis": color_analysis,
                "description": description
            })

        total_frames = len(analysed_keyframes)
        color_distribution_percentages = {
            color: round(count/total_frames * 100, 2)
            for color, count in color_distribution.items()
        }

        theme_analysis = self.get_theme_analysis(color_distribution_percentages)

        overall_result = {
            "primary_colors": sorted(
                [(color, count/total_frames) for color, count in color_distribution.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "color_distribution": color_distribution_percentages,
            "theme_analysis": theme_analysis,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_keyframes_analysed": total_frames
        }

        with open("visual_analysis_result.json", "w", encoding="utf-8") as f:
            json.dump({"frames": analysed_keyframes, "overall": overall_result}, f, indent=2)

        return analysed_keyframes