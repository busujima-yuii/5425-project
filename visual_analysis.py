import os
import cv2
import json
import base64
import numpy as np
import time
from openai import OpenAI, RateLimitError
from typing import List, Dict, Tuple
from datetime import datetime
from PIL import Image
from util import (
    analyse_image_colors,
    compress_image,
    compute_ssim,
    extract_json_from_response,
)
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

api_key = ""

class VisualAnalysis:
    def __init__(self, output_folder="keyframes"):
        self.output_folder = output_folder
        self.client = OpenAI(api_key=api_key)
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_keyframes(
        self,
        video_path: str,
        ssim_threshold=0.5,
        min_frame_gap=240,
        max_keyframes=None,
        scoring="ssim",
    ) -> List[Dict]:
        print("\n🖼️ Extracting keyframes...")
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

            if frame_idx % 10 != 0:
                frame_idx += 1
                continue

            print(f"Extracting keyframes{frame_idx}...")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_score = compute_ssim(prev_gray, gray_frame)

            if (
                ssim_score < ssim_threshold
                and (frame_idx - last_frame_idx) >= min_frame_gap
            ):
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                score = (
                    1.0 - ssim_score if scoring == "ssim" else 0
                )  # more different means more interesting
                candidates.append(
                    {
                        "timestamp": round(timestamp, 2),
                        "frame": frame,
                        "score": score,
                        "frame_idx": frame_idx,
                    }
                )
                last_frame_idx = frame_idx

            prev_gray = gray_frame
            prev_frame = frame
            frame_idx += 1

        cap.release()

        print("\n🖼️ Sampling keyframes...")

        if max_keyframes and len(candidates) > max_keyframes:
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[
                :max_keyframes
            ]
            candidates = sorted(
                candidates, key=lambda x: x["frame_idx"]
            )  # restore chronological order

        keyframes = []
        for i, cand in enumerate(candidates):
            fname = f"keyframe_{i}.jpg"
            print(f"\n🖼️ Saving {fname}...")
            fpath = os.path.join(self.output_folder, fname)
            cv2.imwrite(fpath, cand["frame"])
            keyframes.append(
                {
                    "timestamp": cand["timestamp"],
                    "filepath": fpath,
                    "frame": cand["frame"],
                }
            )

        return keyframes

    def extract_keyframes_pyscenedetect(
        self,
        video_path: str,
        threshold: float = 30.0,
        ssim_threshold=0.5,
        max_keyframes: int = None,
    ) -> List[Dict]:
        from datetime import timedelta
        import cv2

        print("\n🖼️ Extracting keyframes...")

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)

        keyframes = []
        os.makedirs(self.output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        for i, (start, end) in enumerate(scene_list):
            timestamp = start.get_seconds()
            frame_num = start.get_frames()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = cap.read()
            if not success:
                continue

            filename = f"keyframe_{i}.jpg"
            filepath = os.path.join(self.output_folder, filename)
            # print(f"\n🖼️ Saving {filename}...")
            # cv2.imwrite(filepath, frame)
            keyframes.append(
                {"timestamp": round(timestamp, 2), "filepath": filepath, "frame": frame}
            )

        cap.release()

        print("\n🖼️ Sampling keyframes...")

        prev_frame = keyframes[0]["frame"]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        candidates = []

        for idx in range(len(keyframes)):
            frame = keyframes[idx]["frame"]

            print(f"Extracting keyframes{idx}/{len(keyframes)}...")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_score = compute_ssim(prev_gray, gray_frame)

            timestamp = keyframes[idx]["timestamp"]
            score = 1.0 - ssim_score
            candidates.append(
                {
                    "timestamp": round(timestamp, 2),
                    "frame": frame,
                    "score": score,
                    "frame_idx": idx,
                }
            )

            prev_gray = gray_frame
            prev_frame = frame

        if max_keyframes and len(candidates) > max_keyframes:
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[
                :max_keyframes
            ]
            candidates = sorted(
                candidates, key=lambda x: x["frame_idx"]
            )  # restore chronological order

        keyframes = []
        for i, cand in enumerate(candidates):
            fname = f"keyframe_{i}.jpg"
            print(f"\n🖼️ Saving {fname}...")
            fpath = os.path.join(self.output_folder, fname)
            cv2.imwrite(fpath, cand["frame"])
            keyframes.append(
                {
                    "timestamp": cand["timestamp"],
                    "filepath": fpath,
                    "frame": cand["frame"],
                }
            )

        return keyframes

    def describe_image_with_gpt(self, image_path: str) -> str:
        compressed = compress_image(image_path)
        with open(compressed, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        for i in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You describe visual content in clear and short English.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please describe the main objects and scene in this image.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=150,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                wait_time = 10 * (2**i)
                print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return "[ERROR] Description failed after retries."

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

        Respond in JSON format only, no markdowns. Format in the following JSON schema 
        {{
            "theme": text,
            "mood": text,
            "genre": text,
            "confidence_score": number,
            "reasoning": text
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis expert specialising in color theory and visual storytelling.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            cleaned = extract_json_from_response(response.choices[0].message.content)
            # print(cleaned)
            return json.loads(cleaned)
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return {
                "theme": "unknown",
                "mood": "unknown",
                "genre": "unknown",
                "confidence_score": 0,
                "reasoning": "Error in theme analysis",
            }

    def get_frame_analysis(self, relevant_word: str, image_path: str) -> Dict:
        relevant_word = (
            "No specific target word. Just analyse freely."
            if relevant_word is None or relevant_word == ""
            else relevant_word
        )
        prompt = f"""Based on the frame picture in a video, analyze the potential theme, mood, and genre.
        Provide your analysis in JSON format with the following fields:
        - theme: main theme or subject matter
        - mood: emotional atmosphere
        - genre: potential video genre
        - score: 0-1 indicating confidence in the analysis
        - relevant_score: 0-1 indicating score of relevance with the target word
        - description: describe the main objects and scene in this image.
        - reasoning: brief explanation of your analysis

        Target word: \"{relevant_word}\"

        Respond in JSON format only, no markdowns. Format in the following JSON schema 
        {{
            "theme": text,
            "mood": text,
            "genre": text,
            "confidence_score": number,
            "relevant_score": number,
            "description": text,
            "reasoning": text
        }}
        """

        compressed = compress_image(image_path)
        with open(compressed, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        for i in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a video analysis expert specialising in color theory and visual storytelling.",
                        },
                        # {"role": "user", "content": prompt},
                        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                cleaned = extract_json_from_response(
                    response.choices[0].message.content
                )
                # print(cleaned)
                return json.loads(cleaned)
            except RateLimitError as e:
                wait_time = 10 * (2**i)
                print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Error in OpenAI API call: {str(e)}")

        return {
            "theme": "unknown",
            "mood": "unknown",
            "genre": "unknown",
            "confidence_score": 0,
            "relevant_score": 0,
            "main_idea": "unknown",
            "reasoning": "Error in theme analysis",
        }

    def analyse_video(
        self, video_path: str, relevant_word: str, max_frames=30
    ) -> List[Dict]:
        cache_path = os.path.join(self.output_folder, "keyframes.json")
        result_path = os.path.join(self.output_folder, "visual_analysis_result.json")
        if os.path.exists(cache_path):
            print("🔁 Loading keyframes from cache...")
            with open(cache_path, "r", encoding="utf-8") as f:
                keyframes = json.load(f)
        else:
            keyframes = self.extract_keyframes_pyscenedetect(
                video_path, max_keyframes=max_frames
            )
            # remove frame ndarray before saving
            serializable_keyframes = [
                {k: v for k, v in kf.items() if k != "frame"} for kf in keyframes
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable_keyframes, f, indent=2)

        if os.path.exists(result_path):
            print("🔁 Loading results from cache...")
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                return result["frames"]
        else:
            analysed_keyframes = []
            color_distribution = {}

            for i in range(len(keyframes)):
                if i % 5 == 0:  # 每5帧暂停10秒
                    time.sleep(10)
                print(f"\n🖼️ Analysis keyframes{i}...")
                kf = keyframes[i]

                if "frame" in kf:
                    frame = kf["frame"]
                else:
                    frame = cv2.imread(kf["filepath"])

                color_analysis = analyse_image_colors(frame)
                description = self.get_frame_analysis(
                    image_path=kf["filepath"], relevant_word=relevant_word
                )

                for color_info in color_analysis["dominant_colors"]:
                    color = color_info["color"]
                    if color not in color_distribution:
                        color_distribution[color] = 0
                    color_distribution[color] += 1

                analysed_keyframes.append(
                    {
                        "timestamp": kf["timestamp"],
                        "image_path": kf["filepath"],
                        "color_analysis": color_analysis,
                        "description": description,
                    }
                )

            total_frames = len(analysed_keyframes)
            color_distribution_percentages = {
                color: round(count / total_frames * 100, 2)
                for color, count in color_distribution.items()
            }

            theme_analysis = self.get_theme_analysis(color_distribution_percentages)

            overall_result = {
                "primary_colors": sorted(
                    [
                        (color, count / total_frames)
                        for color, count in color_distribution.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "color_distribution": color_distribution_percentages,
                "theme_analysis": theme_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_keyframes_analysed": total_frames,
            }

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"frames": analysed_keyframes, "overall": overall_result},
                    f,
                    indent=2,
                )

            return analysed_keyframes
