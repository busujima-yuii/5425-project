import whisper

model = whisper.load_model("base")

# 视频转录为以下数据形式, start和end为起止时间, 均为整数. text是带punctuation的转录文本
# [{"start" : ... , "end" : ... , "text": ...}]
# e.g.
# [
#     {"start" : 0 , "end" : 5 , "text": "Today is a sunny day!"},
#     {"start" : 5 , "end" : 10 , "text": "Today is a rainy day!"}
# ]
#

def transcribe(path):
    result = model.transcribe(path)
    return [
        {"start":round(seg["start"]),
         "end": round(seg["end"]),
         "text": seg["text"]
        }
        for seg in result["segments"]
    ]