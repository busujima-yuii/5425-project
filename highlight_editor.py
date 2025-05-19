import json
from collections import Counter
from moviepy.editor import VideoFileClip, concatenate_videoclips


def load_segments(path="audio_analysis_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["segments"]


def load_visuals(path="visual_analysis_result.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["frames"]


def collect_statistics(segments):
    emotion_counter = Counter()
    theme_counter = Counter()
    for seg in segments:
        analysis = seg.get("analysis", {})
        emotion_counter[analysis.get("emotional_tone", "unknown")] += 1
        theme_counter[analysis.get("theme", "unknown")] += 1
    return emotion_counter, theme_counter


def find_matching_segments(segments, visuals, score_threshold=0.5):
    matched = []
    covered_audio = 0
    covered_visual = 0

    for seg in segments:
        if seg.get("relevant_score", 0) >= score_threshold:
            matched.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": seg["relevant_score"],
                "source": "audio"
            })
            covered_audio += 1

    for frame in visuals:
        if frame.get("relevant_score", 0) >= score_threshold:
            ts = frame["timestamp"]
            matched.append({
                "start": ts - 1.0,
                "end": ts + 2.0,
                "score": frame["relevant_score"],
                "source": "visual"
            })
            covered_visual += 1

    coverage_info = {
        "matched_audio_segments": covered_audio,
        "matched_visual_frames": covered_visual,
        "score_threshold": score_threshold
    }

    return matched, coverage_info


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


def print_statistics():
    segments = load_segments()
    emotion_counts, theme_counts = collect_statistics(segments)
    print("\n📊 Emotion Distribution:", dict(emotion_counts))
    print("📊 Theme Distribution:", dict(theme_counts))


def run_keyword_highlight(video_path, score_threshold=0.5, max_duration=60.0):
    print(f"🔍 Filtering segments by score >= {score_threshold}")

    audio_segments = load_segments()
    visuals = load_visuals()
    matched_segments, coverage_info = find_matching_segments(audio_segments, visuals, score_threshold)

    merged_segments = merge_close_segments(matched_segments)
    limited_segments = limit_total_duration(merged_segments, max_duration=max_duration)

    print(f"✂️ Selected {len(limited_segments)} merged segments (limit {max_duration}s)")
    print("📈 Coverage Info:", coverage_info)

    extract_clips(video_path, limited_segments)


if __name__ == "__main__":
    video_path = "your_video.mp4"  # Replace with your actual video
    print_statistics()
    run_keyword_highlight(video_path, score_threshold=0.5, max_duration=60.0)
