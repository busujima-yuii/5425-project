import whisper

model = whisper.load_model("base")

def transcribe(path):
    result = model.transcribe(path)
    return [
        {"start":round(seg["start"]),
         "end": round(seg["end"]),
         "text": seg["text"]
        }
        for seg in result["segments"]
    ]