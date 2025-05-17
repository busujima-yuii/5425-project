import openai
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

def extract_audio(video_path, output_path="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path)
    print(f"🎧 Audio extracted to: {output_path}")
    return output_path

def transcribe_audio_whisper(audio_path):
    print("📝 Transcribing with Whisper...")
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe(model="whisper-1", file=f)
    text = transcript["text"]
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("✅ Transcript saved to transcript.txt")
    return text

def summarize_text_gpt4(text):
    print("🧠 Summarizing with GPT-4...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes transcripts into natural, clear summaries."},
        {"role": "user", "content": f"Please summarize the following transcript in 3–5 sentences:\n\n{text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300
    )
    summary = response["choices"][0]["message"]["content"].strip()
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("✅ Summary saved to summary.txt")
    return summary

def split_audio_to_chunks(audio_path, chunk_length=30 * 1000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    total_ms = len(audio)
    for i in range(0, total_ms, chunk_length):
        chunk = audio[i:i + chunk_length]
        start_sec = i / 1000
        end_sec = min((i + chunk_length), total_ms) / 1000
        fname = f"chunk_{int(start_sec)}_{int(end_sec)}.wav"
        chunk.export(fname, format="wav")
        chunks.append((fname, start_sec, end_sec))
    return chunks

def transcribe_chunks_with_timestamps(chunks):
    openai.api_key = "YOUR_API_KEY"
    transcript_with_timestamps = []

    for fname, start, end in chunks:
        with open(fname, "rb") as f:
            print(f"⏱️ Transcribing: {fname}")
            resp = openai.Audio.transcribe("whisper-1", f)
            transcript_with_timestamps.append({
                "start": start,
                "end": end,
                "text": resp["text"].strip()
            })

    return transcript_with_timestamps

