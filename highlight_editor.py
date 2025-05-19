import json
from collections import Counter
from moviepy.editor import VideoFileClip, concatenate_videoclips


def load_segments(path="audio_analysis_results.json"):
    if path[len(path)-1] != "/":
        path = path+"/"
    if "audio_results" not in path:
        path = path+"audio_results/"
    if "audio_analysis_results.json" not in path:
        path = path+"audio_analysis_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["segments"]

def load_visuals(path="visual_analysis_result.json"):
    if path[len(path)-1] != "/":
        path = path+"/"
    if "visual_results" not in path:
        path = path+"visual_results/"
    if "visual_analysis_result.json" not in path:
        path = path+"visual_analysis_result.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["frames"]


def collect_audio_statistics(segments):
    emotion_counter = Counter()
    theme_counter = Counter()
    for seg in segments:
        analysis = seg.get("analysis", {})
        emotion_counter[analysis.get("emotional_tone", "unknown")] += 1
        theme_counter[analysis.get("theme", "unknown")] += 1
    return emotion_counter, theme_counter


def match_keywords_in_text(text, keywords):
    if not keywords:
        return True
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)


def find_matching_segments(segments, visuals, score_threshold=0.5, emotion_filter=None, theme_filter=None, keywords=None):
    matched = []
    coverage = {
        "matched_audio_segments": 0,
        "matched_visual_frames": 0,
        "score_threshold": score_threshold,
        "emotion_filter": emotion_filter,
        "theme_filter": theme_filter,
        "keywords": keywords
    }

    for seg in segments:
        score = seg.get("relevant_score", 0)
        analysis = seg.get("analysis", {})
        emo = analysis.get("emotional_tone", "unknown")
        theme = analysis.get("theme", "unknown")
        text = seg.get("transcript", "") + seg.get("summary", "") + " ".join(analysis.get("key_points", []))

        if score < score_threshold:
            continue
        if emotion_filter and emo not in emotion_filter:
            continue
        if theme_filter and theme not in theme_filter:
            continue
        if not match_keywords_in_text(text, keywords):
            continue

        matched.append({
            "start": seg["start"],
            "end": seg["end"],
            "score": score,
            "source": "audio",
            "emotion": emo,
            "theme": theme
        })
        coverage["matched_audio_segments"] += 1

    for frame in visuals:
        score = frame.get("relevant_score", 0)
        desc = frame.get("description", "")
        if score < score_threshold:
            continue
        if not match_keywords_in_text(desc, keywords):
            continue

        ts = frame["timestamp"]
        matched.append({
            "start": ts - 1.0,
            "end": ts + 2.0,
            "score": score,
            "source": "visual"
        })
        coverage["matched_visual_frames"] += 1

    return matched, coverage


def merge_close_segments(segments, max_gap=3.0):
    if not segments:
        return []
    segments.sort(key=lambda x: x["start"])
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
            last["score"] = max(last.get("score", 0), seg.get("score", 0))
            last["source"] += f",{seg.get('source', '')}"
        else:
            merged.append(seg)
    return merged


def limit_total_duration(segments, max_duration=60.0):
    segments.sort(key=lambda x: x.get("score", 0), reverse=True)
    total = 0.0
    result = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if total + dur <= max_duration:
            result.append(seg)
            total += dur
        else:
            break
    return result


def extract_clips(video_path, segments, output_path="highlight.mp4"):
    video = VideoFileClip(video_path)
    clips = [video.subclip(s["start"], s["end"]) for s in segments]
    final = concatenate_videoclips(clips)
    final.write_videofile(output_path)

    # Save metadata
    metadata = {
        "clips": segments,
        "total_duration": sum(s["end"] - s["start"] for s in segments)
    }
    with open("highlight_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def print_statistics(file_path):
    segments = load_segments(file_path)
    visuals = load_visuals(file_path)
    emotion_counts, theme_counts = collect_audio_statistics(segments)
    print("\n📊 Emotion Distribution:", dict(emotion_counts))
    print("📊 Theme Distribution:", dict(theme_counts))


def run_keyword_highlight(video_path, file_path, score_threshold=0.5, max_duration=60.0,
                          emotion_filter=None, theme_filter=None, keywords=None):
    print(f"🔍 Filtering segments by: score >= {score_threshold}, emotion = {emotion_filter}, theme = {theme_filter}, keywords = {keywords}")

    audio_segments = load_segments(file_path)
    visuals = load_visuals(file_path)
    matched_segments, coverage_info = find_matching_segments(
        audio_segments, visuals,
        score_threshold=score_threshold,
        emotion_filter=emotion_filter,
        theme_filter=theme_filter,
        keywords=keywords
    )

    merged_segments = merge_close_segments(matched_segments)
    limited_segments = limit_total_duration(merged_segments, max_duration=max_duration)

    print(f"✂️ Selected {len(limited_segments)} merged segments (limit {max_duration}s)")
    print("📈 Coverage Info:", coverage_info)

    extract_clips(video_path, limited_segments)


if __name__ == "__main__":
    video_path = "E:/5425/Sense8.S01E03.mp4"
    output_path = "E:/5425/"
    print_statistics()
    
    run_keyword_highlight(
        video_path,
        score_threshold=0.5,
        max_duration=60.0,
        emotion_filter=["tense", "joyful"],
        theme_filter=["war", "family"],
        keywords=["escape", "reunion"]
    )