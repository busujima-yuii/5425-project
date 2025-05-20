
# 🎬 Multimodal Video Highlight Generator

This project provides a multimodal video highlight generation tool based on audio and visual analysis. It uses OpenAI GPT models and Whisper transcription to identify and extract the most meaningful segments from full-length videos and automatically generate a concise highlight clip.

##  Author: Page(Peizhi) Zhu, Beichen Yu, Lynda Li

## 📦 Project Structure

| File | Description |
|------|-------------|
| `gui_app.py` | Main interactive interface built with Gradio |
| `highlight_editor.py` | Core logic: scoring, filtering, segment merging, clip generation |
| `audio_analysis.py` | Whisper-based transcription and GPT-based audio semantic/emotional analysis |
| `visual_analysis.py` | Frame extraction and GPT-based visual content + color theme analysis |
| `test.py` | Sample test entry to run the analysis pipeline |
| `util.py` | Utility functions for image processing, merging, JSON extraction, etc. |

## 🚀 Key Features

- 🎧 **Audio Analysis**: Transcription with timestamps, GPT-powered emotional and thematic analysis
- 🖼️ **Visual Analysis**: Keyframe extraction, color theme classification, visual content description
- 🤖 **GPT Integration**: Uses `gpt-4o-mini` or `gpt-3.5-turbo` for all intelligent interpretation tasks
- 📊 **Multidimensional Filtering**: Filter by emotion, theme, genre, dialogue type, and more
- ✂️ **Automatic Highlighting**: Extracts and merges top-scoring segments into a final highlight video
- 💻 **User-Friendly GUI**: Full workflow managed via Gradio-based interface

## 🧠 Highlights

- ✅ **Keyword-Optional**: If no keyword is entered, the system will fall back to confidence-based scoring
- 🔍 **Multimodal Matching**: Combines GPT relevance score and embedding-based semantic similarity
- 🔁 **Segment Merging**: Merges nearby clips to form coherent highlights
- ⏱️ **Duration Limit**: Total output duration can be capped (e.g., 60 seconds)
- 🛡️ **Fallback Logic**: If filters are too strict, system auto-relaxes constraints to ensure results

## 🛠 Installation

```bash
pip install -r requirements.txt
```

### Key dependencies:
- `gradio`
- `openai`
- `moviepy`
- `faster-whisper`
- `sentence-transformers`
- `opencv-python`
- `scenedetect`

## 💻 How to Launch the GUI

```bash
python gui_app.py
```

Visit [http://localhost:7860](http://localhost:7860) in your browser.

## 📂 Workflow (GUI)

1. Select or upload a video
2. Optionally enter a keyword (e.g., “danger”)
3. Click **Start Analysis** to perform both audio and visual analysis
4. View summaries and filter statistics
5. Select desired filters (theme, mood, genre, emotion, dialogue type)
6. Click **Generate Highlight** to produce the final video

## 📁 Output Directory Structure

```
video_outputs/
└── <video_name>/
    ├── audio_results/
    │   └── audio_analysis_results.json
    ├── visual_results/
    │   └── visual_analysis_result.json
    ├── highlight.mp4
    └── highlight_metadata.json
```

## 🔧 CLI Mode (for batch testing)

```bash
python highlight_editor.py
```

Edit the `__main__` section in `highlight_editor.py` to set `video_path` and `output_path`.

## 📩 Contact

For questions, collaboration, or issues, please reach out COMP5425 Project Group9: beyu0824 pzhu0521 chdu0298 yuli4954
