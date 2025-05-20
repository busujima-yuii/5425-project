from audio_analysis import AudioAnalysis
from visual_analysis import VisualAnalysis
import util
import json
import os


def test_audio_analysis(video_path: str, output_path: str, relevant_word=None):
    print("\n🎧 Running Audio Analysis...")
    audio_analysis = AudioAnalysis(output_folder=output_path + "/audio_results")

    result = audio_analysis.run_audio_analysis(
        video_path=video_path, output_path=output_path, relevant_word=relevant_word
    )
    # summary = result["full_summary"]
    # stats = result["distribution_summary"]
    # print("\n🧠 Full Summary:")
    # print(summary)
    # print("\n📊 Distribution Summary:")
    # print(stats)


def test_visual_analysis(video_path: str, output_path: str, relevant_word=None):
    print("\n🖼️ Running Visual Analysis...")
    visual = VisualAnalysis(output_folder=output_path + "/visual_results")

    # Step 1: Analyse keyframes and image descriptions
    results = visual.analyse_video(video_path, relevant_word=relevant_word)

    # print("\n🖼️ Sample Keyframe Description:")
    # for item in results[:2]:  # print first 2
    #     print(f"⏱ {item['timestamp']}s → {item['description']}")

    # print("\n🎨 Color Analysis:")
    # for color in results[0]["color_analysis"]["dominant_colors"]:
    #     print(f"- {color['color']} ({color['percentage']}%)")


if __name__ == "__main__":
    video_file = "E:/5425/Sense8.S01E03.mp4"
    output_path = "E:/5425"
    # test_audio_analysis(video_file, output_path)
    # test_visual_analysis(video_file,output_path)
