import os
from clip import clip
from indexer import build_index
from search import search_segments
from transcribe import transcribe
import shutil

from util import merge_segments
from audio_summariser import AudioSummarizer

filename = "sciam_0002.mp3"
indexdir = "indexdir"

segments = transcribe(filename)
merged = merge_segments(segments, min_seg_duration=2.0)

# build_index(segments)
# hits = search_segments("quote", limit=5)
# for h in hits:
#     print(h)
#
# if os.path.exists(indexdir):
#     shutil.rmtree(indexdir)

content = merged
for i in range(len(content)):
    line = content[i]
    print(line)

# trial for ffmpeg clipping
#     clipped_bytes = clip(filename, line['start'], line['end'])
#     with open("clip" + str(i) + ".mp4", "wb") as out:
#         out.write(clipped_bytes)

"""Example usage of the AudioSummariser"""
# Get API key from environment variable
# api_key = os.getenv("openai_api_key")
# if not api_key:
#     print("Please set the OPENAI_API_KEY environment variable")

# summariser = AudioSummariser(api_key)

# try:
#     overall_analysis, segment_analysis = summariser.summarise_audio(filename)
    
#     print("\nOverall Analysis:")
#     print(json.dumps(overall_analysis, indent=2))
    
#     print("\nSegment Analysis:")
#     for segment in segment_analysis:
#         print(f"\nTime: {segment['start']}-{segment['end']}s")
#         print(f"Text: {segment['text']}")
#         print("Analysis:", json.dumps(segment['analysis'], indent=2))
        
# except Exception as e:
#     print(f"Error processing audio: {str(e)}")