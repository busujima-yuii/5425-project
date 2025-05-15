import os
from clip import clip
from indexer import build_index
from search import search_segments
from transcribe import transcribe
import shutil

from util import merge_segments

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