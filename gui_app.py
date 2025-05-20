import gradio as gr
import os
import shutil
import json
from collections import Counter
from test import test_audio_analysis, test_visual_analysis
from highlight_editor import run_keyword_highlight

state = {"video_path": "", "output_path": "", "summary": "", "stats": {}, "filters": {}}


def load_video_path(path):
    if not os.path.exists(path):
        return "❌ File does not exist", None
    state["video_path"] = path
    state["output_path"] = os.path.join(
        "video_outputs", os.path.splitext(os.path.basename(path))[0]
    )
    os.makedirs(state["output_path"], exist_ok=True)
    return f"✅ Ready: {path}", path


def handle_file_selection(uploaded_file):
    # 从临时路径获取原始名称
    temp_path = uploaded_file.name
    filename = os.path.basename(temp_path)

    # 模拟原始路径（仅限演示；你也可以让用户修改 Textbox）
    user_visible_path = os.path.abspath(temp_path)  # 也可以替换为用户上传前的路径

    # 设置状态为此路径
    state["video_path"] = temp_path
    state["output_path"] = os.path.join("video_outputs", os.path.splitext(filename)[0])
    os.makedirs(state["output_path"], exist_ok=True)

    return temp_path, temp_path  # -> Textbox + Video preview


def select_video_traditional(video_file):
    state["video_path"] = video_file
    state["output_path"] = os.path.dirname(video_file)
    return f"✅ Selected: {video_file}", state["output_path"]


def select_video(uploaded_file):
    original_path = uploaded_file.name
    filename = os.path.basename(original_path)
    folder_name = os.path.splitext(filename)[0]

    output_dir = os.path.join("video_outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    target_path = os.path.join(output_dir, filename)
    shutil.copy(original_path, target_path)

    state["video_path"] = target_path
    state["output_path"] = output_dir

    return f"✅ Copied to: {target_path}", target_path, output_dir


def build_summary_text(audio_path, visual_path):
    summary_lines = []

    # Audio
    if os.path.exists(audio_path):
        with open(audio_path, "r", encoding="utf-8") as f:
            audio_data = json.load(f)
            stats = audio_data.get("distribution_summary", {})
            summary_lines.append("🎧 Audio Summary:")
            summary_lines.append(f"- Segments: {stats.get('total_segments', '?')}")
            top_type = list(stats.get("dialogue_type_distribution", {}).keys())[:3]
            top_emotion = list(stats.get("emotional_tone_distribution", {}).keys())[:3]
            top_theme = list(stats.get("theme_distribution", {}).keys())[:3]
            summary_lines.append(f"- Top Dialogue Types: {', '.join(top_type)}")
            summary_lines.append(f"- Top Emotions: {', '.join(top_emotion)}")
            summary_lines.append(f"- Top Themes: {', '.join(top_theme)}")
            summary_lines.append("")

    # Visual
    if os.path.exists(visual_path):
        with open(visual_path, "r", encoding="utf-8") as f:
            visual_data = json.load(f)
            frames = visual_data.get("frames", [])
            summary_lines.append("🎞️ Visual Summary:")
            summary_lines.append(f"- Frames: {len(frames)}")

            from collections import Counter

            mood_counter = Counter()
            genre_counter = Counter()
            theme_counter = Counter()
            for frame in frames:
                desc = frame.get("description", {})
                mood_counter[desc.get("mood", "unknown")] += 1
                genre_counter[desc.get("genre", "unknown")] += 1
                theme_counter[desc.get("theme", "unknown")] += 1

            summary_lines.append(
                f"- Top Moods: {', '.join([m for m, _ in mood_counter.most_common(3)])}"
            )
            summary_lines.append(
                f"- Top Genres: {', '.join([g for g, _ in genre_counter.most_common(3)])}"
            )
            summary_lines.append(
                f"- Top Themes: {', '.join([t for t, _ in theme_counter.most_common(3)])}"
            )

    return "\n".join(summary_lines)


def analyze_video(keyword):
    if not state["video_path"]:
        return (
            "⚠️ Please select a video file first.",
            "",
            {},
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    test_audio_analysis(state["video_path"], state["output_path"], keyword)
    test_visual_analysis(state["video_path"], state["output_path"], keyword)

    audio_result_path = os.path.join(
        state["output_path"], "audio_results", "audio_analysis_results.json"
    )
    visual_result_path = os.path.join(
        state["output_path"], "visual_results", "visual_analysis_result.json"
    )

    summary = ""
    stats = {}
    counters = {
        "emotion": Counter(),
        "theme": Counter(),
        "mood": Counter(),
        "genre": Counter(),
        "dialogue_type": Counter(),
    }

    print("Done Loading/Analysing Results")

    if os.path.exists(audio_result_path):
        with open(audio_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            summary = data.get("full_summary", "")
            stats = data.get("distribution_summary", {})
            for seg in data.get("segments", []):
                a = seg.get("analysis", {})
                counters["emotion"][a.get("emotional_tone", "unknown")] += 1
                counters["theme"][a.get("theme", "unknown")] += 1
                counters["dialogue_type"][a.get("dialogue_type", "unknown")] += 1

    if os.path.exists(visual_result_path):
        with open(visual_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for frame in data.get("frames", []):
                d = frame.get("description", {})
                counters["theme"][d.get("theme", "unknown")] += 1
                counters["mood"][d.get("mood", "unknown")] += 1
                counters["genre"][d.get("genre", "unknown")] += 1

    filters = {
        key: [item for item, _ in counters[key].most_common(10)] for key in counters
    }
    state["summary"] = summary
    state["stats"] = stats
    state["filters"] = {k: sorted(list(v)) for k, v in filters.items()}

    stats_summary = build_summary_text(audio_result_path, visual_result_path)
    return (
        "✅ Analysis Complete",
        summary,
        stats_summary,
        gr.update(choices=sorted(filters["emotion"])),
        gr.update(choices=sorted(filters["theme"])),
        gr.update(choices=sorted(filters["mood"])),
        gr.update(choices=sorted(filters["genre"])),
        gr.update(choices=sorted(filters["dialogue_type"])),
    )


def generate_highlight(
    score_threshold,
    duration,
    emotion,
    theme,
    mood,
    genre,
    dialogue_type,
    keyword_input,
    use_relevance,
):
    if not state["video_path"]:
        return "⚠️ No video selected."

    # actual_threshold = score_threshold if use_relevance else 0.0

    output_path = run_keyword_highlight(
        video_path=state["video_path"],
        file_path=state["output_path"],
        score_threshold=score_threshold,
        max_duration=duration,
        emotion_filter=emotion or None,
        theme_filter=theme or None,
        mood_filter=mood or None,
        genre_filter=genre or None,
        dialogue_type_filter=dialogue_type or None,
        keywords=[keyword_input] if keyword_input else None,
        use_relevant=use_relevance,
    )
    return output_path


with gr.Blocks() as demo:
    gr.Markdown("🎬 **Multimodal Video Highlight GUI**")

    # with gr.Row():
    #     with gr.Column():
    #         video_input = gr.Video(label="Select Video", interactive=True)
    #         # video_path_box = gr.Textbox(
    #         #     label="📂 Video File Path",
    #         #     placeholder="E:/myvideo.mp4",
    #         #     interactive=True,
    #         # )
    #         # video_preview = gr.Video(label="📺 Preview Video")
    #     with gr.Column():
    #         output_folder = gr.Textbox(label="Output Folder", interactive=False)
    #         status = gr.Textbox(label="Status")
    with gr.Row():
        video_input = gr.Video(label="Select Video", interactive=True)
        with gr.Column():
            output_folder = gr.Textbox(label="Output Folder", interactive=False)
            status = gr.Textbox(label="Status")

    keyword_input = gr.Textbox(label="Enter Keyword (e.g. danger)")
    analyze_btn = gr.Button("Start Analysis")

    with gr.Row():
        summary_output = gr.Textbox(label="Full Summary")
        stats_output = gr.Textbox(label="Distribution Summary")

    emotion = gr.CheckboxGroup(choices=[], label="Emotion")
    theme = gr.CheckboxGroup(choices=[], label="Theme")
    mood = gr.CheckboxGroup(choices=[], label="Mood")
    genre = gr.CheckboxGroup(choices=[], label="Genre")
    dialogue_type = gr.CheckboxGroup(choices=[], label="Dialogue Type")

    score_threshold = gr.Slider(0, 1, value=0.5, label="Score Threshold")
    max_duration = gr.Slider(10, 300, value=60, step=10, label="Max Total Duration (s)")

    # file_picker.change(
    #     fn=handle_file_selection,
    #     inputs=file_picker,
    #     outputs=[video_path_box, video_preview]
    # )
    # load_btn = gr.Button("📂 Load Video")

    # load_btn.click(
    #     fn=load_video_path, inputs=video_path_box, outputs=[status, video_preview]
    # )

    video_input.change(
        fn=select_video_traditional,
        inputs=
        # [
        video_input,
        # video_path_box],
        outputs=[
            status,
            #  video_preview,
            output_folder,
        ],
    )

    analyze_btn.click(
        fn=analyze_video,
        inputs=keyword_input,
        outputs=[
            status,
            summary_output,
            stats_output,
            emotion,
            theme,
            mood,
            genre,
            dialogue_type,
        ],
    )

    matching_keyword_input = gr.Textbox(
        label="Enter Keyword for matching (e.g. danger)"
    )
    use_relevance = gr.Checkbox(label="Use Relevant Score in Matching", value=True)
    generate_btn = gr.Button("🎞️ Generate Highlight")
    highlight_output = gr.Video(label="Output Highlight")

    generate_btn.click(
        fn=generate_highlight,
        inputs=[
            score_threshold,
            max_duration,
            emotion,
            theme,
            mood,
            genre,
            dialogue_type,
            matching_keyword_input,
            use_relevance,
        ],
        outputs=highlight_output,
    )

demo.launch(server_name="0.0.0.0", share=False)
