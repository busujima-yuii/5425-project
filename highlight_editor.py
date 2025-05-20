import json
from collections import Counter
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Embedding Utility ---------------- #


def build_embeddings_from_visual_json(json_path: str) -> Dict[str, List[float]]:
    with open(json_path, "r", encoding="utf-8") as f:
        visuals = json.load(f)["frames"]
    embeddings = {}
    for frame in visuals:
        desc = frame.get("description", {})
        text = f"{desc.get('description', '')} {desc.get('reasoning', '')}"
        embeddings[str(frame["timestamp"])] = model.encode(text)
    return embeddings


def build_embeddings_from_audio_json(json_path: str) -> Dict[str, List[float]]:
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    embeddings = {}
    for seg in segments:
        analysis = seg.get("analysis", {})
        text = f"{seg.get('summary', '')} {analysis.get('main_idea', '')} {' '.join(analysis.get('key_points', []))}"
        embeddings[f"{seg['start']}-{seg['end']}"] = model.encode(text)
    return embeddings


def ensure_embeddings_loaded(
    json_path: str, cache_path: str, mode: str
) -> Dict[str, List[float]]:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        if mode == "visual":
            embeddings = build_embeddings_from_visual_json(json_path)
        else:
            embeddings = build_embeddings_from_audio_json(json_path)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings


def encode_keywords(keywords: List[str]) -> List[List[float]]:
    return [model.encode(k) for k in keywords]


def compute_semantic_score(text_embedding, keyword_embeddings) -> float:
    scores = [float(util.cos_sim(text_embedding, kw)[0]) for kw in keyword_embeddings]
    return max(scores) if scores else 0.0


# ---------------- Unified Matching ---------------- #


def load_segments(path="audio_analysis_results.json"):
    if path[len(path) - 1] != "/":
        path = path + "/"
    if "audio_results" not in path:
        path = path + "audio_results/"
    if "audio_analysis_results.json" not in path:
        path = path + "audio_analysis_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["segments"], path


def load_visuals(path="visual_analysis_result.json"):
    if path[len(path) - 1] != "/":
        path = path + "/"
    if "visual_results" not in path:
        path = path + "visual_results/"
    if "visual_analysis_result.json" not in path:
        path = path + "visual_analysis_result.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["frames"], path


def collect_statistics(segments):
    emotion_counter = Counter()
    theme_counter = Counter()
    for seg in segments:
        analysis = seg.get("analysis", {})
        emotion_counter[analysis.get("emotional_tone", "unknown")] += 1
        theme_counter[analysis.get("theme", "unknown")] += 1
    return emotion_counter, theme_counter


def collect_visual_statistics(frames):
    theme_counter = Counter()
    mood_counter = Counter()
    genre_counter = Counter()
    for frame in frames:
        analysis = frame.get("description", {})
        theme_counter[analysis.get("theme", "unknown")] += 1
        mood_counter[analysis.get("mood", "unknown")] += 1
        genre_counter[analysis.get("genre", "unknown")] += 1
    return theme_counter, mood_counter, genre_counter


def match_keywords_in_text(text, keywords):
    if not keywords:
        return True
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)


def find_matching_segments(
    segments,
    visuals,
    score_threshold=0.5,
    emotion_filter=None,
    theme_filter=None,
    keywords=None,
):
    matched = []
    coverage = {
        "matched_audio_segments": 0,
        "matched_visual_frames": 0,
        "score_threshold": score_threshold,
        "emotion_filter": emotion_filter,
        "theme_filter": theme_filter,
        "keywords": keywords,
    }

    for seg in segments:
        score = seg.get("relevant_score", 0)
        analysis = seg.get("analysis", {})
        emo = analysis.get("emotional_tone", "unknown")
        theme = analysis.get("theme", "unknown")
        text = (
            seg.get("transcript", "")
            + seg.get("summary", "")
            + " ".join(analysis.get("key_points", []))
        )

        if score < score_threshold:
            continue
        if emotion_filter and emo not in emotion_filter:
            continue
        if theme_filter and theme not in theme_filter:
            continue
        if not match_keywords_in_text(text, keywords):
            continue

        matched.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "score": score,
                "source": "audio",
                "emotion": emo,
                "theme": theme,
            }
        )
        coverage["matched_audio_segments"] += 1

    for frame in visuals:
        score = frame.get("relevant_score", 0)
        desc = frame.get("description", "")
        if score < score_threshold:
            continue
        if not match_keywords_in_text(desc, keywords):
            continue

        ts = frame["timestamp"]
        matched.append(
            {"start": ts - 1.0, "end": ts + 2.0, "score": score, "source": "visual"}
        )
        coverage["matched_visual_frames"] += 1

    return matched, coverage


def find_matching_segments_with_embeddings(
    segments,
    visuals,
    audio_json_path,
    visual_json_path,
    score_threshold=0.5,
    keywords=None,
    emotion_filter=None,
    theme_filter=None,
    mood_filter=None,
    genre_filter=None,
    dialogue_type_filter=None,
    min_results=5,
    use_relevant=True,
):
    from copy import deepcopy

    keywords = keywords or []
    keyword_embeddings = encode_keywords(keywords)

    audio_embeddings = ensure_embeddings_loaded(
        audio_json_path, audio_json_path + ".emb.pkl", "audio"
    )
    visual_embeddings = ensure_embeddings_loaded(
        visual_json_path, visual_json_path + ".emb.pkl", "visual"
    )

    def apply_filters(item, filters):
        for key, allowed in filters.items():
            value = item.get(key)
            if allowed and value not in allowed:
                return False
        return True

    def score_audio(seg, emb_vec, use_relevant=True):
        analysis = seg.get("analysis", {})
        base_score = analysis.get("relevant_score", 0)
        text_score = compute_semantic_score(emb_vec, keyword_embeddings)
        if use_relevant == False:
            return text_score
        return 0.6 * base_score + 0.4 * text_score

    def score_visual(desc, emb_vec, use_relevant=True):
        base_score = desc.get("relevant_score", 0)
        text_score = compute_semantic_score(emb_vec, keyword_embeddings)
        if use_relevant == False:
            return text_score
        return 0.6 * base_score + 0.4 * text_score

    filters = {
        "theme": theme_filter,
        "mood": mood_filter,
        "genre": genre_filter,
        "emotion": emotion_filter,
        "dialogue_type": dialogue_type_filter,
    }

    def collect_matches(filters_to_use, use_relevant=True):
        matched = []
        for seg in segments:
            analysis = seg.get("analysis", {})
            emb_vec = audio_embeddings.get(f"{seg['start']}-{seg['end']}")
            if emb_vec is None:
                continue
            score = score_audio(seg, emb_vec, use_relevant=use_relevant)
            if score < score_threshold:
                continue
            if not apply_filters(
                analysis,
                {
                    "theme": filters_to_use.get("theme"),
                    "emotion": filters_to_use.get("emotion"),
                    "dialogue_type": filters_to_use.get("dialogue_type"),
                },
            ):
                continue
            matched.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "score": score,
                    "source": "audio",
                    "theme": analysis.get("theme"),
                    "emotion": analysis.get("emotional_tone"),
                    "dialogue_type": analysis.get("dialogue_type"),
                }
            )

        for frame in visuals:
            desc = frame.get("description", {})
            ts = str(frame["timestamp"])
            emb_vec = visual_embeddings.get(ts)
            if emb_vec is None:
                continue
            score = score_visual(desc, emb_vec, use_relevant=use_relevant)
            if score < score_threshold:
                continue
            if not apply_filters(
                desc,
                {
                    "theme": filters_to_use.get("theme"),
                    "mood": filters_to_use.get("mood"),
                    "genre": filters_to_use.get("genre"),
                },
            ):
                continue
            matched.append(
                {
                    "start": float(ts) - 1.0,
                    "end": float(ts) + 2.0,
                    "score": score,
                    "source": "visual",
                    "theme": desc.get("theme"),
                    "mood": desc.get("mood"),
                    "genre": desc.get("genre"),
                }
            )
        return matched

    if not keywords:
        matched = []
        for seg in segments:
            analysis = seg.get("analysis", {})
            matched.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "score": (
                        analysis.get("relevant_score", 1.0)
                        if use_relevant
                        else analysis.get("score", 1.0)
                    ),
                    "source": "audio",
                    "theme": analysis.get("theme"),
                    "emotion": analysis.get("emotional_tone"),
                    "dialogue_type": analysis.get("dialogue_type"),
                }
            )

        for frame in visuals:
            desc = frame.get("description", {})
            ts = frame["timestamp"]
            matched.append(
                {
                    "start": ts - 1.0,
                    "end": ts + 2.0,
                    "score": (
                        analysis.get("relevant_score", 1.0)
                        if use_relevant
                        else analysis.get("score", 1.0)
                    ),
                    "source": "visual",
                    "theme": desc.get("theme"),
                    "mood": desc.get("mood"),
                    "genre": desc.get("genre"),
                }
            )
    else:
        matched = collect_matches(filters, use_relevant=use_relevant)

    fallback_used = False
    if len(matched) < min_results:
        fallback_used = True
        matched = collect_matches({k: None for k in filters}, use_relevant=use_relevant)

    coverage = {
        "matched_audio_segments": len([m for m in matched if m["source"] == "audio"]),
        "matched_visual_frames": len([m for m in matched if m["source"] == "visual"]),
        "score_threshold": score_threshold,
        "keywords": keywords,
        "fallback_used": fallback_used,
    }

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
    os.makedirs(output_path, exist_ok=True)
    data_path = output_path
    if "highlight.mp4" not in output_path:
        output_path = output_path + "/highlight.mp4"

    video = VideoFileClip(video_path)
    max_duration = video.duration

    clips = []
    for s in segments:
        start = max(0, float(s["start"]))
        end = min(float(s["end"]), max_duration)

        if end <= start or (end - start) < 0.2:  # 防止空片段或过短
            print(f"⛔ Skipping invalid segment: {start} - {end}")
            continue

        try:
            clip = video.subclip(start, end)
            clips.append(clip)
        except Exception as e:
            print(f"❌ Error extracting clip {start}-{end}: {e}")
            continue

    if not clips:
        print("⚠️ No valid clips to export.")
        return

    final = concatenate_videoclips(clips)
    final.write_videofile(output_path)

    video.reader.close()
    if video.audio:
        video.audio.reader.close_proc()

    for c in clips:
        c.close()
    final.close()

    # Save metadata
    metadata = {
        "clips": segments,
        "total_duration": sum(s["end"] - s["start"] for s in segments),
    }
    with open(data_path + "/highlight_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def print_statistics(file_path):
    segments, _ = load_segments(file_path)
    visuals, _ = load_visuals(file_path)

    emotion_counts, theme_counts = collect_statistics(segments)
    v_theme, v_mood, v_genre = collect_visual_statistics(visuals)

    def print_top_statistics(counter: Counter, label: str, emoji: str, top_k: 5):
        print(f"\n{emoji} {label} Top {top_k}:")
        for name, count in counter.most_common(top_k):
            print(f"  {name}: {count}")

    print_top_statistics(emotion_counts, "Emotion Distribution", "📊", top_k=5)
    print_top_statistics(theme_counts, "Audio Theme Distribution", "🧠", top_k=5)
    print_top_statistics(v_theme, "Visual Theme Distribution", "🎨", top_k=5)
    print_top_statistics(v_mood, "Visual Mood Distribution", "😺", top_k=5)
    print_top_statistics(v_genre, "Visual Genre Distribution", "🎬", top_k=5)


def run_keyword_highlight(
    video_path,
    file_path,
    score_threshold=0.5,
    max_duration=60.0,
    emotion_filter=None,
    theme_filter=None,
    mood_filter=None,
    genre_filter=None,
    dialogue_type_filter=None,
    keywords=None,
    use_relevant=True,
):
    print(
        f"🔍 Filtering segments by: score >= {score_threshold}, emotion = {emotion_filter}, theme = {theme_filter}, mood = {mood_filter}, keywords = {keywords}"
    )

    audio_segments, audio_path = load_segments(file_path)
    visuals, visual_path = load_visuals(file_path)
    matched_segments, coverage_info = find_matching_segments_with_embeddings(
        audio_segments,
        visuals,
        audio_json_path=audio_path,
        visual_json_path=visual_path,
        score_threshold=score_threshold,
        emotion_filter=emotion_filter,
        theme_filter=theme_filter,
        mood_filter=mood_filter,
        genre_filter=genre_filter,
        dialogue_type_filter=dialogue_type_filter,
        keywords=keywords,
        use_relevant=use_relevant,
    )

    merged_segments = merge_close_segments(matched_segments)
    limited_segments = limit_total_duration(merged_segments, max_duration=max_duration)

    print(f"✂️ Selected {len(limited_segments)} merged segments (limit {max_duration}s)")
    print("📈 Coverage Info:", coverage_info)

    extract_clips(video_path, limited_segments, file_path)


if __name__ == "__main__":
    video_path = "E:/5425/Sense8.S01E03.mp4"
    output_path = "E:/5425/"
    print_statistics(output_path)
    run_keyword_highlight(
        video_path,
        output_path,
        score_threshold=0.5,
        max_duration=60.0,
        emotion_filter=None,
        theme_filter=None,
        mood_filter=None,
        keywords=["danger"],
    )
