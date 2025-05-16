import os
from clip import clip
from indexer import build_index
from search import search_segments
from transcribe import transcribe
import shutil

from util import merge_segments
from audio_summariser import AudioSummariser
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

"""AudioSummariser"""
# Get API key from environment variable
openai_api_key = "api-key placeholder"

summariser = AudioSummariser(openai_api_key)

 # Path to the Peppa Pig video
video_path = "test/peppa_pig.MOV"

try:
    overall_analysis, segment_analysis = summariser.summarise_audio(video_path)
    
    print("\nOverall Analysis:")
    print(json.dumps(overall_analysis, indent=2))
    
    # Segment Anlysis
    # print("\nSegment Analysis:")
    # for segment in segment_analysis:
    #     print(f"\nTime: {segment['start']}-{segment['end']}s")
    #     print(f"Text: {segment['text']}")
    #     print("Analysis:", json.dumps(segment['analysis'], indent=2))

    # Save the segment analysis to a JSON file# Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output filename
    output_filename = f"{video_name}_results.json"
    output_path = results_dir / output_filename
    
    # Save the analysis to a JSON file
    output = {
        "overall_analysis": overall_analysis,
        "segment_analysis": segment_analysis
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"\nAnalysis has been saved to {output_path}")
        
except Exception as e:
    print(f"Error processing audio: {str(e)}")
