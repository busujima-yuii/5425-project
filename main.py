import os
from clip import clip
from indexer import build_index
from search import search_segments
from transcribe import transcribe
import shutil
from pathlib import Path

from util import merge_segments
from audio_analyser import AudioAnalyser
from hue_analyser import HueAnalyser
import json


# filename = "sciam_0002.mp3"
# indexdir = "indexdir"

# segments = transcribe(filename)
# merged = merge_segments(segments, min_seg_duration=2.0)

# build_index(segments)
# hits = search_segments("quote", limit=5)
# for h in hits:
#     print(h)
#
# if os.path.exists(indexdir):
#     shutil.rmtree(indexdir)

# content = merged
# for i in range(len(content)):
#     line = content[i]
#     print(line)

# trial for ffmpeg clipping
#     clipped_bytes = clip(filename, line['start'], line['end'])
#     with open("clip" + str(i) + ".mp4", "wb") as out:
#         out.write(clipped_bytes)

"""Audio and Hue Analysis"""
# Get API key from environment variable
openai_api_key = "api-key placeholder"

# Initialize analysers
audio_analyser = AudioAnalyser(openai_api_key)
hue_analyser = HueAnalyser(openai_api_key)

# Path to the video
video_path = "video_placeholder"

try:
    # Perform audio analysis
    overall_audio_analysis, segment_analysis = audio_analyser.analyse_audio(video_path)
    
    # Perform hue analysis
    overall_hue_analysis, frame_analysis = hue_analyser.analyze_video(video_path)
    
    print("\nOverall Audio Analysis:")
    print(json.dumps(overall_audio_analysis, indent=2))
    
    print("\nOverall Hue Analysis:")
    print(json.dumps(overall_hue_analysis, indent=2))

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output filename
    output_filename = f"{video_name}_results.json"
    output_path = results_dir / output_filename
    
    # Save the analysis to a JSON file
    output = {
        "audio_analysis": {
            "overall": overall_audio_analysis,
            "segments": segment_analysis
        },
        "hue_analysis": {
            "overall": overall_hue_analysis,
            "frames": frame_analysis
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"\nAnalysis has been saved to {output_path}")
        
except Exception as e:
    print(f"Error processing video: {str(e)}")
